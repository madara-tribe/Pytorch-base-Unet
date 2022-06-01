import sys, os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.utils import data
from cfg import Cfg
from utils.meta_dataloder import MetaDataLoader
from unet import UNet
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity # SSIM

def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    return val

# https://emotionexplorer.blog.fc2.com/blog-entry-380.html
def create_data_loader(config):
    val_dst = MetaDataLoader(config.x_img, config.y_meta, val=True)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch_size, shuffle=None, num_workers=0, pin_memory=True)
    print("Val set: %d" % len(val_dst))
    return val_loader, val_dst
 
def predict(config, device, path):
    model = UNet(n_channels=3, n_classes=config.classes, bilinear=config.bilinear)
    model.to(device)
    model.load_state_dict(torch.load(path))
    print("loaded trained model")
    val_loader, val_dst = create_data_loader(config)
    nums = 0
    meta_mse, meta_ssim = 0, 0
    model.eval()
    with tqdm(total=len(val_dst)) as pbar:
        for i, batch in tqdm(enumerate(val_loader)):
            val_x_img = batch['image'].to(device=device, dtype=torch.float32)
            # predict
            pred_meta = model(val_x_img)
            # torch to numpy
            pred_meta = pred_meta[0].detach().cpu().numpy()
            pred_meta = (pred_meta * 255).reshape(224, 224).astype(np.uint8)
            
            val_y_meta = batch["mask"].detach().cpu().numpy()
            val_y_meta = (val_y_meta*255).astype(np.uint8).reshape(224, 224)
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

