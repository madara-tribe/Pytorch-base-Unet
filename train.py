import logging
import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from cfg import Cfg
from utils.meta_dataloder import MetaDataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNet
from utils.optimizer import create_optimizer
from utils.loss import MetaLoss

class Trainer:
    def __init__(self, config, num_workers, pin_memory=True):
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_data_loader(self, config):
        x_img = np.load("data/x_img.npy")
        y_rgb = np.load("data/y_rgb.npy")
        y_shape = np.load("data/y_shape.npy")
        val_x_img, val_y_rgb, val_y_shape = np.flip(x_img), np.flip(y_rgb), np.flip(y_shape)
        self.train_dst = MetaDataLoader(x_img, y_rgb, y_shape)
        self.val_dst = MetaDataLoader(val_x_img, val_y_rgb, val_y_shape)

    
        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch_size, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))
    


    def validate(self, model, global_step, epoch, device):
        interval_valloss, val_rgb, val_shape = 0, 0, 0
        nums = 0
        model.eval()
        with torch.no_grad():
            for i, (val_x_img, val_y_rgb, val_y_shape) in tqdm(enumerate(self.val_loader)):
                print("validating .....")
                val_x_img, val_y_rgb, val_y_shape = val_x_img.permute(0, 3, 1, 2), val_y_rgb.permute(0, 3, 1, 2), val_y_shape #.permute(0, 3, 1, 2)
                val_x_img, val_y_rgb, val_y_shape = val_x_img.to(device=device), val_y_rgb.to(device=device), val_y_shape.to(device=device)
                # predict valid
                pred_rgb_, pred_shape_ = model(val_x_img)

                # val loss update
                val_rgb_loss = self.criterion.rgb_loss(val_y_rgb, pred_rgb_)
                val_shape_loss = self.criterion.shape_loss(val_y_shape, pred_shape_)
                val_loss = val_rgb_loss + val_shape_loss
                interval_valloss += val_loss.item()
                val_rgb += val_rgb_loss.item()
                val_shape += val_shape_loss.item()
                nums += 1

            self.tfwriter.add_scalar('valid/interval_loss', interval_valloss/nums, global_step)
            self.tfwriter.add_scalar('valid/val_rgb', val_rgb/nums, global_step)
            self.tfwriter.add_scalar('valid/val_shape', val_shape/nums, global_step)
            print("Epoch %d, Itrs %d, valid_Loss=%f, valid_rgb=%f, val_shape=%f" % (epoch, global_step, interval_valloss/nums, val_rgb/nums, val_shape/nums))
    
    
    def train(self, config, device, weight_path=None):
        model = UNet(n_channels=3, n_classes=config.classes, bilinear=config.bilinear)
        
        self.create_data_loader(config)
        # (Initialize logging)
        logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Learning rate:   {config.lr}
            Training size:   {len(self.train_dst)}
            Validation size: {len(self.val_dst)}
        ''')
        
        # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer, scheduler = create_optimizer(model, config)
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path, map_location=device))
        #if torch.cuda.device_count() > 1:
           # model = nn.DataParallel(model)
        model.to(device)
        self.criterion = MetaLoss()

        global_step = 0
        # 5. Begin training
        for epoch in range(1, config.epochs+1):
            interval_loss, rgb, shape = 0, 0, 0
            model.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for x_img, y_rgb, y_shape in self.train_loader:
                    x_img = x_img.permute(0, 3, 1, 2) #y_rgb.permute(0, 3, 1, 2), y_shape#.permute(0, 3, 1, 2)
                    x_img, y_rgb, y_shape = x_img.to(device=device), y_rgb.to(device=device), y_shape.to(device=device)
                    # predict
                    pred_rgb, pred_shape = model(x_img)

                    # loss update
                    rgb_loss = self.criterion.rgb_loss(y_rgb, pred_rgb)
                    shape_loss = self.criterion.shape_loss(y_shape, pred_shape) 
                    loss = rgb_loss + shape_loss
                    interval_loss += loss.item()
                    rgb +=rgb_loss.item()
                    shape += shape_loss.item()  
                    loss.backward()
                    optimizer.step()

                    pbar.update()
                    global_step += 1
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    if global_step % 10 == 0:
                        self.tfwriter.add_scalar('train/train_loss', interval_loss/10, global_step)
                        
                        self.tfwriter.add_scalar('train/train_loss', rgb/10, global_step)
                        self.tfwriter.add_scalar('train/train_loss', shape/10, global_step)
                        print("Epoch %d, Itrs %d, Loss=%f, rgb_loss=%f, shape_loss=%f" %
                            (epoch, global_step, interval_loss/10, rgb_loss.item()/10, shape_loss.item()/10))
                        interval_loss = rgb = shape = 0.0

                    if global_step % config.val_interval == 0:
                        self.validate(model, global_step, epoch, device)

            if config.save_checkpoint:
                Path(config.ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), config.ckpt_dir + "/"+ 'checkpoint_epoch{}.pth'.format(epoch))
                logging.info(f'Checkpoint {epoch} saved!')




if __name__ == '__main__':
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    weight_path = None
    trainer = Trainer(cfg, num_workers=1, pin_memory=True)
    try:
        trainer.train(cfg, device=device, weight_path=weight_path)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        raise



