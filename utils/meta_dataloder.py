import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np
import cv2
W = H = 224

class MetaDataLoader(data.Dataset):
    def __init__(self,
                 x_img, x_meta, 
                 y_rgb, y_meta, 
                 width=W,
                 height=H,
                 transform=None):
        
        self.transform = transform
        self.width = width
        self.height = height
        self.x_img, self.x_meta = x_img[:100], x_meta[:100]
        self.y_rgb, self.y_meta = y_rgb[:100], y_meta[:100]
        assert (len(self.x_img) == len(self.y_rgb))
        assert (len(self.x_meta) == len(self.y_meta))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x_img = self.x_img[index].astype(np.float32)/255
        x_meta = self.x_meta[index].astype(np.float32)/255
        
        y_rgb = self.y_rgb[index].astype(np.float32)/255
        y_meta = self.y_meta[index].astype(np.float32)/255
        #if self.transform is not None:
            #augment = self.transform(image=img, mask=target)
            #img, target = augment['image'], augment['mask']
        return x_img, x_meta, y_rgb, y_meta
    def __len__(self):
        return len(self.x_img)
