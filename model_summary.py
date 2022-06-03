import torch
from unet.custom_unet1 import UNet
from torchsummary import summary

if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channel=3, out_channel=3)
    model.to(device)
    summary(model, (3, 224, 224))

