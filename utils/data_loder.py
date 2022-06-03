import os
import sys
import torch
import torch.utils.data as data
import numpy as np
import cv2
import glob

W = H = 224

class BasicDataLoader(data.Dataset):
    def __init__(self,
                 x_img, y_meta,
                 width: int = W,
                 height: int = H,
                 val=None):
        self.width = width
        self.height = height
        # train image
        x_imgs = os.listdir(x_img)
        x_imgs.sort()
        x_imgs = [os.path.join(x_img, path) for path in x_imgs]
        y_metas = os.listdir(y_meta)
        y_metas.sort()
        y_metas = [os.path.join(y_meta, path) for path in y_metas]
        if val:
            self.y_metas = y_metas[100:150]
            self.x_imgs = x_imgs[100:150]
        else:
            self.y_metas = y_metas[:100]
            self.x_imgs = x_imgs[:100]
        assert (len(self.y_metas) == len(self.x_imgs))

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
        #x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = x_img.transpose(2, 0, 1).astype(np.float32)/255
        x_img = torch.as_tensor(x_img.copy()).float().contiguous()

        y_meta = cv2.imread(self.y_metas[index])
        y_meta = cv2.cvtColor(y_meta, cv2.COLOR_BGR2LAB)
        y_meta = y_meta.transpose(2, 0, 1).astype(np.float32)/255
        y_meta = torch.as_tensor(y_meta.copy()).float().contiguous()

        return x_img, y_meta
