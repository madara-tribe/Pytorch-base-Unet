import sys, os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils import data
from cfg import Cfg
from utils.data_loder import BasicDataLoader
from utils.utils import ToGray
#from unet import UNet
from unet.custom_unet2 import UNet
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity # SSIM
H = W = 256
def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    return val

# https://emotionexplorer.blog.fc2.com/blog-entry-380.html
def create_data_loader(config):
    val_dst = BasicDataLoader(config.x_img, config.x_img, val=True)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch, shuffle=None, num_workers=0, pin_memory=True)
    print("Val set: %d" % len(val_dst))
    return val_loader, val_dst

def predict(config, device, path):
    model = UNet(in_channel=3, out_channel=config.classes)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader, val_dst = create_data_loader(config)
    nums = 0
    meta_mse, meta_ssim = 0, 0
    model.eval()
    with tqdm(total=len(val_dst)) as pbar:
        for i, (x, y) in tqdm(enumerate(val_loader)):
            val_x_img = x.to(device=device, dtype=torch.float32)
            # predict
            pred_meta = model(val_x_img)
            # torch to numpy
            pred_meta = pred_meta[0].detach().cpu().numpy()
            pred_meta = (pred_meta * 255).reshape(H, W, 3).astype(np.float32)
            pred_meta = ToGray(pred_meta)
            val_y_meta = y[0].detach().cpu().numpy()
            val_y_meta = (val_y_meta*255).reshape(H, W, 3).astype(np.float32)
            val_y_meta = ToGray(val_y_meta)
            meta_mse_score = mean_squared_error(val_y_meta, pred_meta)
            meta_ssim_score = measurement(structural_similarity, img1=val_y_meta, img2=pred_meta)
            meta_mse += meta_mse_score
            meta_ssim += meta_ssim_score
            nums += 1
            pbar.update()
        
        print('  MSE :', meta_mse/nums)
        print('  SSIM:', meta_ssim/nums)
            
            
if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch13.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)

