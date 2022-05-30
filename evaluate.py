import sys, os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils import data
from cfg import Cfg
from utils.meta_dataloder import MetaDataLoader
from unet import UNet
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity # SSIM

def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    return val

# https://emotionexplorer.blog.fc2.com/blog-entry-380.html
def create_data_loader(config):
    x_img = np.load("data/x_img.npy")
    x_meta = np.load("data/x_meta.npy")
    y_rgb = np.load("data/y_rgbmeta.npy")
    y_meta = np.load("data/y_meta.npy")
    val_x_img, val_x_meta, val_y_rgb, val_y_meta = x_img, x_meta, y_rgb, y_meta
    
    val_dst = MetaDataLoader(val_x_img, val_x_meta, val_y_rgb, val_y_meta)


    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch_size, shuffle=None, num_workers=0)
    print("Val set: %d" % len(val_dst))
    return val_loader
    
def predict(config, device, path):
    model = UNet(n_channels=3, n_classes=config.classes, bilinear=config.bilinear)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader = create_data_loader(config)
    nums = 0
    rgb_mse, rgb_ssim = 0, 0
    shape_mse, shape_ssim = 0, 0
    model.eval()
    with tqdm(total=100) as pbar:
        for i, (val_x_img, val_x_meta, val_y_rgb, val_y_meta) in tqdm(enumerate(val_loader)):
            val_x_img, val_x_meta =val_x_img.permute(0, 3, 1, 2), val_x_meta.permute(0, 3, 1, 2)
            val_x_img = val_x_img.to(device=device, dtype=torch.float32)
            val_x_meta = val_x_meta.to(device=device, dtype=torch.float32)
            val_y_rgb = val_y_rgb.to(device=device, dtype=torch.float32)
            val_y_meta = val_y_meta.to(device=device, dtype=torch.float32)
            # predict
            pred_rgb_, pred_meta_ = model(val_x_img, val_x_meta)
            # torch to numpy
            pred_rgb_ = pred_rgb_[0].detach().cpu().numpy()
            pred_meta_ = pred_meta_[0].detach().cpu().numpy()
            pred_rgb_ = (pred_rgb_ * 255).transpose(1, 2, 0).astype(np.uint8)
            pred_meta_ = (pred_meta_ * 255).transpose(1, 2, 0).astype(np.uint8).reshape(224,224)
            
            val_y_rgb = val_y_rgb[0].detach().cpu().numpy()
            val_y_meta = val_y_meta[0].detach().cpu().numpy()
            val_y_rgb = (val_y_rgb*255).astype(np.uint8)
            val_y_meta =(val_y_meta*255).astype(np.uint8).reshape(224, 224)
            #print(i, rgb_mse, pred_rgb_.shape, val_y_rgb.shape, val_y_meta.shape, pred_meta_.shape) 
            #rgb_mse_score = mean_squared_error(val_y_rgb, pred_rgb_)
            #rgb_ssim_score = measurement(structural_similarity, img1=val_y_rgb, img2=pred_rgb_)
            shape_mse_score = mean_absolute_error(val_y_meta, pred_meta_)
            shape_ssim_score = measurement(structural_similarity, img1=val_y_meta, img2=pred_meta_)
            #rgb_mse += rgb_mse_score
            #rgb_ssim += rgb_ssim_score
            shape_mse += shape_mse_score
            shape_ssim += shape_ssim_score
            nums += 1
            pbar.update()
        
        #print('  RGB MSE :', rgb_mse/nums)
        #print('  RGB SSIM:', rgb_ssim/nums)
        print('  SHAPE MSE :', shape_mse/nums)
        print('  SHAPE SSIM:', shape_ssim/nums)
            
            
if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch20.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)
