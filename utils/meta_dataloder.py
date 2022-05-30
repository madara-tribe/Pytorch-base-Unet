import os
import sys
import torch.utils.data as data
import numpy as np
import cv2

W = H = 224

class MetaDataLoader(data.Dataset):
    def __init__(self,
                 x_img, y_rgb, y_shape, 
                 width=W,
                 height=H,
                 transform=None):
        self.transform = transform
        self.width = width
        self.height = height
        self.x_img, self.y_rgb, self.y_shape = x_img[:100], y_rgb[:100], y_shape[:100]
        assert (len(self.x_img) == len(self.y_rgb))
        assert (len(self.y_shape) == len(self.y_rgb))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        x_img = self.x_img[index].astype(np.float32)/255
        y_rgb = self.y_rgb[index].astype(np.float32)
        y_shape = self.y_shape[index].astype(np.float32)/255
        #if self.transform is not None:
            #augment = self.transform(image=img, mask=target)
            #img, target = augment['image'], augment['mask']
        return x_img, y_rgb, y_shape

    def __len__(self):
        return len(self.x_img)

