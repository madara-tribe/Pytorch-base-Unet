import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2
import glob

W = H = 224

class MetaDataLoader(data.Dataset):
    def __init__(self,
                 x_img, y_rgb, 
                 width: int = W,
                 height: int = H,
                 val=None):
        self.width = width
        self.height = height
        # train image
        x_imgs = os.listdir(x_img)
        x_imgs.sort()
        x_imgs = [os.path.join(x_img, path) for path in x_imgs]
        y_rgbs = os.listdir(y_rgb)
        y_rgbs.sort()
        y_rgbs = [os.path.join(y_rgb, path) for path in y_rgbs]
        if val:
            self.y_rgbs = y_rgbs[100:150]
            self.x_imgs = x_imgs[100:150]
        else:
            self.y_rgbs = y_rgbs[:100]
            self.x_imgs = x_imgs[:100]
        assert (len(self.y_rgbs) == len(self.y_rgbs))

    def __len__(self):
        return len(self.x_imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x_img = cv2.imread(self.x_imgs[index])
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = x_img.transpose(2, 0, 1).astype(np.float32)/255
       
        y_rgb = cv2.imread(self.y_rgbs[index])
        y_rgb = y_rgb.transpose(2, 0, 1).astype(np.float32)/255
        return {
            'image': torch.as_tensor(x_img.copy()).float().contiguous(),
            'mask1': torch.as_tensor(y_rgb.copy()).float().contiguous(),
            #'mask2': torch.as_tensor(y_shape.copy()).long().contiguous()
        }

