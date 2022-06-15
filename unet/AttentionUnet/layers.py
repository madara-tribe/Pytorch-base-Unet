import torch
from torch import nn
from torch.nn import functional as F


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(conv2d, self).__init__()
        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU())
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
         x = self.conv(x)
         x = self.dropout(x)
         return x

class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv2d, self).__init__()
        self.deconv = nn.Sequential(
                      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                      nn.BatchNorm2d(out_channels),
                      nn.GELU())
        def forward(self, x):
            x = self.deconv(x)
            return x



class attention_block(nn.Module):
    def __init__(self, size, bn=False):
        super(attention_block, self).__init__()
        self.conv1 = nn.Conv2d(size, size, kernel_size=(1,1), stride=(1,1))# padding='valid')
        self.bn1 = nn.BatchNorm2d(size)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(size, size, kernel_size=(1,1), stride=(1,1)) #padding='valid')
        self.bn2 = nn.BatchNorm2d(size)
        self.sigmoid = nn.Sigmoid()
    def forward(self, downc, upc):
        g = self.conv1(downc)
        g = self.bn1(g)
        x = self.conv1(upc)
        x = self.bn1(x)
        psi = torch.add(g, x)  
        psi = self.gelu(psi)
        psi = self.conv2(psi)
        psi = self.bn2(psi)
        psi = self.sigmoid(psi)
        return torch.mul(upc, psi)
