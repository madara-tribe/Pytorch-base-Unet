import torch 
import torch.nn as nn
import torch.nn.functional as F


def contracting_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
                nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )
    return block
def expansive_block(in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
                nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(mid_channel),
                nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                )
        return  block
    
def final_block(in_channels, mid_channel, out_channels, kernel_size=3):
    block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            nn.Sigmoid(),
            )
    return  block

class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = contracting_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = contracting_block(128, 256)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = contracting_block(256, 512)
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        self.bottleneck = nn.Sequential(
                            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(1024),
                            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(1024),
                            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        # Decode
        self.conv_decode4 = expansive_block(1024, 512, 256)
        self.conv_decode3 = expansive_block(512, 256, 128)
        self.conv_decode2 = expansive_block(256, 128, 64)
        self.final_layer = final_block(128, 64, out_channel)
        
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        #print(encode_block1.shape)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4  = self.conv_maxpool4(encode_block4)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # Decode
        #print(x.shape, encode_block1.shape, encode_block2.shape, encode_block3.shape, encode_pool3.shape, bottleneck1.shape)
        #print('Decode Block 3')
        #print(bottleneck1.shape, encode_block3.shape)
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer3 = self.conv_decode4(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=True)
        #print(decode_block3.shape)
        #print('Decode Block 2')
        cat_layer2 = self.conv_decode3(decode_block3)
        #print(cat_layer2.shape, encode_block2.shape)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        #print(cat_layer1.shape, encode_block1.shape)
        #print('Final Layer')
        #print(cat_layer1.shape, encode_block1.shape)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        #print(decode_block1.shape)
        final_layer = self.final_layer(decode_block1)
        #print(final_layer.shape)
        return  final_layer
