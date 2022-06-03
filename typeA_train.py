import logging
from pathlib import Path
import sys, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchsummary import summary
from tqdm import tqdm
from cfg import Cfg
from utils.data_loder import BasicDataLoader
from torch.utils.tensorboard import SummaryWriter
#from unet import UNet
from unet.custom_unet import UNet
from utils.optimizer import create_optimizer
from utils import utils, loss



class Trainer:
    def __init__(self, config, num_workers, pin_memory=True):
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.criterion = nn.MSELoss()

    def create_data_loader(self, config):
        
        self.train_dst = BasicDataLoader(config.x_img, config.x_img, val=None)
        self.val_dst = BasicDataLoader(config.x_img, config.x_img, val=True)
        
        self.train_loader = data.DataLoader(
                self.train_dst, batch_size=config.train_batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)
        self.val_loader = data.DataLoader(
                    self.val_dst, batch_size=config.val_batch, shuffle=None, num_workers=self.num_workers, pin_memory=self.pin_memory)
        print("Train set: %d, Val set: %d" %(len(self.train_dst), len(self.val_dst)))
    

    def validate(self, model, global_step, epoch, device):
        interval_valloss = 0
        nums = 0
        model.eval()
        with torch.no_grad():
            for i, (val_x_img, val_y_meta) in tqdm(enumerate(self.val_loader)):
                print("validating .....")
                val_x_img = val_x_img.to(device=device, dtype=torch.float32)
                val_y_meta = val_y_meta.to(device=device, dtype=torch.float32)
                # predict valid
                pred_meta = model(val_x_img)

                # val loss update
                val_loss = self.criterion(val_y_meta, pred_meta)
                interval_valloss += val_loss.item()
                nums += 1

            self.tfwriter.add_scalar('valid/interval_loss', interval_valloss/nums, global_step)
            print("Epoch %d, Itrs %d, valid_Loss=%f" % (epoch, global_step, interval_valloss/nums))
    
    
    def train(self, config, device, weight_path=None):
        #model = UNet(inc=3, num_cls=config.classes)
        model = UNet(in_channel=3, out_channel=config.classes)
        self.create_data_loader(config)
        # (Initialize logging)
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
        summary(model, (3, 224, 224))

        global_step = 0
        # 5. Begin training
        for epoch in range(1, config.epochs+1):
            interval_loss = 0
            model.train()
            with tqdm(total=int(len(self.train_dst)/config.train_batch), desc=f'Epoch {epoch}/{config.epochs}') as pbar:
                for x_img, y_meta in self.train_loader:
                    x_img = x_img.to(device=device, dtype=torch.float32)
                    y_meta = y_meta.to(device=device, dtype=torch.float32)
                    # predict)
                    pred_meta = model(x_img)
                    if global_step % 3==0:
                        utils.save_in_progress(x_img, y_meta, pred_meta, global_step)
                    # loss update
                    loss = self.criterion(y_meta, pred_meta)
                    interval_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    pbar.update()
                    global_step += 1
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    if global_step % 10 == 0:
                        self.tfwriter.add_scalar('train/train_loss', interval_loss/10, global_step)
                        print("Epoch %d, Itrs %d, Loss=%f" % (epoch, global_step, interval_loss/10))
                        interval_loss = 0.0

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
    trainer = Trainer(cfg, num_workers=4, pin_memory=True)
    try:
        trainer.train(cfg, device=device, weight_path=weight_path)
    except KeyboardInterrupt:
        logging.info('Saved interrupt')
        raise

