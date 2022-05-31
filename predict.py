import sys, os
import numpy as np
import torch
from torch.utils import data
import cv2
from tqdm import tqdm
from cfg import Cfg
from utils.meta_dataloder import MetaDataLoader
from unet import UNet

def create_data_loader(config):
    val_dst = MetaDataLoader(config.x_img, config.y_rgb, val=True)
    val_loader = data.DataLoader(
                val_dst, batch_size=config.val_batch_size, shuffle=None, num_workers=0, pin_memory=True)
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
        for i, batch in tqdm(enumerate(val_loader)):
            print("validating .....")
            val_x_img = batch['image'].to(device=device, dtype=torch.float32)
            #val_y_rgb  = batch['mask1'].to(device=device, dtype=torch.float32)
            # predict
            pred_rgb_ = model(val_x_img)
            # torch to numpy
            pred_rgb_ = pred_rgb_[0].detach().cpu().numpy()
            pred_rgb_ = (pred_rgb_ * 255).transpose(1, 2, 0).reshape(224, 224, 3).astype(np.uint8)
            print(pred_rgb_.shape, pred_rgb_.max(), pred_rgb_.min())
            cv2.imwrite("results/pred_rgb_{}.png".format(nums), pred_rgb_)
            nums += 1

if __name__=="__main__":
    path = "checkpoints/checkpoint_epoch20.pth" #str(sys.argv[0])
    cfg = Cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(cfg, device, path)
