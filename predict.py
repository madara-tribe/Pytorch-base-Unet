import sys, os
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from cfg import Cfg
from utils.data_loder import BasicDataLoader, MetaDataLoader
#from unet import UNet
from unet.custom_unet import UNet

val_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def create_data_loader(config, transform_=None):
    if transform_:
        val_dst = MetaDataLoader(config.x_img, config.x_img, transform=val_transform, valid=True)
    else:
        val_dst = BasicDataLoader(config.x_img, config.x_img, val=True)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch, shuffle=None, num_workers=0, pin_memory=None)
    print("Val set: %d" % len(val_dst))
    return val_loader


def predict(config, device, path):
    #model = UNet(inc=3, num_cls=config.classes) 
    model = UNet(in_channel=3, out_channel=config.classes)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader = create_data_loader(config, transform_=True)
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    nums = 0
    model.eval()
    print("predicting  .....")
    with torch.no_grad():
        for i, (val_x_img, val_y_img) in tqdm(enumerate(val_loader)):
            val_x_img = val_x_img.to(device=device, dtype=torch.float32)
            val_y_rgb  = val_y_img.to(device=device, dtype=torch.float32)
            # predict
            pred_meta = model(val_x_img)
            # torch to numpy
            print(val_x_img.shape, val_y_rgb.shape, pred_meta.shape)
            print(val_x_img.min(), val_x_img.max(), val_y_rgb.min(), val_y_rgb.max(), pred_meta.min(), pred_meta.max())
            pred_meta = pred_meta[0].detach().cpu().numpy()
            pred_meta = (pred_meta * 255).transpose(1, 2, 0).astype(np.float32)
            val_x_img = val_x_img[0].detach().cpu().numpy()
            val_x_img = (val_x_img * 255).transpose(1, 2, 0).astype(np.float32)
            val_y_rgb = val_y_rgb[0].detach().cpu().numpy()
            val_y_rgb = (val_y_rgb * 255).transpose(1, 2, 0).astype(np.float32)
            save_img = np.hstack([val_x_img, val_y_rgb, pred_meta])
            print(save_img.shape, save_img.max(), save_img.min())
            cv2.imwrite("results/pred_rgb_{}.png".format(nums), save_img.astype(np.uint8))
            nums += 1

if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch20.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)
