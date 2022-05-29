""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.cls_conv = FeatureConv(64, 1)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, y):
        x1 = self.inc(x)
        y1 = self.inc(y)

        x2 = self.down1(x1)
        y2 = self.down1(y1)
        
        x3 = self.down2(x2)
        y3 = self.down2(y2)

        x4 = self.down3(x3)
        y4 = self.down3(y3)

        x4_ = torch.cat((x4, y4), dim=3)
        x5 = self.down4(x4_)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        feature = self.cls_conv(x)
        logits = self.outc(x)
        return logits, feature


