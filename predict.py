import sys, os
import numpy as np
import torch
from torch.utils import data
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils import data
from tqdm import tqdm
from cfg import Cfg
from utils.meta_dataloder import MetaDataLoader
from unet import UNet

def create_binary_mask(pred, threshold=0.1):
    markers = np.zeros_like(pred)
    markers[pred < threshold] = 0
    markers[pred > threshold] = 1
    #print(np.unique(markers))
    return markers

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
    if not os.path.exists('results'):
        os.makedirs('results', exist_ok=True)
    nums = 0
    model.eval()
    with torch.no_grad():
        for i, (val_x_img, val_x_meta, val_y_rgb, val_y_meta) in tqdm(enumerate(val_loader)):
            print("validating .....")
            val_x_img = val_x_img.to(device=device, dtype=torch.float32)
            val_x_meta = val_x_meta.to(device=device, dtype=torch.float32)
            val_x_img, val_x_meta =val_x_img.permute(0, 3, 1, 2), val_x_meta.permute(0, 3, 1, 2)
            # predict
            pred_rgb_, pred_meta_ = model(val_x_img, val_x_meta)
            # torch to numpy
            pred_rgb_ = pred_rgb_[0].detach().cpu().numpy()
            pred_meta_ = pred_meta_[0].detach().cpu().numpy()
            pred_rgb_ = (pred_rgb_ * 255).transpose(1, 2, 0).astype(np.uint8)
            pred_meta_ = (pred_meta_ * 255).transpose(1, 2, 0).astype(np.uint8)
            #pred_meta_ = create_binary_mask(pred_meta_, threshold=0.2)
            pred_rgb_ = pred_rgb_.reshape(224, 224, 3)
            pred_meta_ = pred_meta_.reshape(224, 224)
            
            print(pred_rgb_.shape, pred_meta_.shape)
            print(pred_meta_.max(), pred_meta_.min(), pred_rgb_.max(), pred_rgb_.min())
            cv2.imwrite("results/pred_meta_{}.png".format(nums), pred_meta_)
            cv2.imwrite("results/pred_rgb_{}.png".format(nums), pred_meta_)
            nums += 1

if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch20.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)
