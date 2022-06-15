import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import conv2d, deconv2d, attention_block


class UNet(nn.Module):    
    def __init__(self, inc, num_cls, start_fm):
        super(UNet, self).__init__()
        
        #(Double) Convolution 1        
        self.double_conv1 = conv2d(inc, start_fm)
        #Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 2
        self.double_conv2 = conv2d(start_fm, start_fm * 2)
        #Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 3
        self.double_conv3 = conv2d(start_fm * 2, start_fm * 4)
        #Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 4
        self.double_conv4 = conv2d(start_fm * 4, start_fm * 8)
        #Max Pooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 5
        self.double_conv5 = conv2d(start_fm * 8, start_fm * 16)
        
        self.convtranas6 = nn.ConvTranspose2d(start_fm * 16, start_fm * 8, 2, 2)
        self.deconv6 = deconv2d(start_fm * 8, start_fm * 8)
        self.atten_block6 = attention_block(start_fm * 8)
        self.conv6 = conv2d(start_fm * 16, start_fm * 8)

        self.convtranas7 = nn.ConvTranspose2d(start_fm * 8, start_fm * 4, 2, 2)
        self.deconv7 = deconv2d(start_fm * 4, start_fm * 4)
        self.atten_block7 = attention_block(start_fm * 4)
        self.conv7 = conv2d(start_fm * 8, start_fm * 4)

        self.convtranas8 = nn.ConvTranspose2d(start_fm * 4, start_fm * 2, 2, 2)
        self.deconv8 = deconv2d(start_fm * 2, start_fm * 2)
        self.atten_block8 = attention_block(start_fm * 2)
        self.conv8 = conv2d(start_fm * 4, start_fm * 2)
        
        self.convtranas9 = nn.ConvTranspose2d(start_fm * 2, start_fm, 2, 2)
        self.deconv9 = deconv2d(start_fm, start_fm)
        self.atten_block9 = attention_block(start_fm)
        self.conv9 = conv2d(start_fm * 2, start_fm)
        
        # One by One Conv
        self.one_by_one = nn.Conv2d(start_fm, num_cls, kernel_size=3, padding=1)
        self.final_act = nn.Sigmoid()
        
        
    def forward(self, inputs):
        # 1
        conv1 = self.double_conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        # 2
        conv2 = self.double_conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        # 3
        conv3 = self.double_conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # 4
        conv4 = self.double_conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
          
        # center
        conv5 = self.double_conv5(maxpool4)
        
        # 6
        conv5 = self.convtranas6(conv5)
        up6 = self.deconv6(conv5)
        conv6 = self.atten_block6(up6, conv4)
        up6 = torch.cat([up6, conv6], 1)
        ex_conv6 = self.conv6(up6)
        
        # 7
        ex_conv6 = self.convtranas7(ex_conv6)
        up7 = self.deconv7(ex_conv6)
        conv7 = self.atten_block7(up7, conv3)
        up7 = torch.cat([up7, conv7], 1)
        ex_conv7 = self.conv7(up7)
  
        # 8
        ex_conv7 = self.convtranas8(ex_conv7)
        up8 = self.deconv8(ex_conv7)
        conv8 = self.atten_block8(up8, conv2)
        up8 = torch.cat([up8, conv8], 1)
        ex_conv8 = self.conv8(up8)
       
        # 9
        ex_conv8 = self.convtranas9(ex_conv8)
        up9 = self.deconv9(ex_conv8)
        conv9 = self.atten_block9(up9, conv1)
        up9 = torch.cat([up9, conv9], 1)
        ex_conv9 = self.conv9(up9)
        
        one_by_one = self.one_by_one(ex_conv9)
        x = self.final_act(one_by_one)
        return x

if __name__=="__main__":
    start_fm = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    H = W = 224
    model = UNet(inc=3, num_cls=1, start_fm=start_fm)
    inp = torch.rand((1, 3, 224, 224), dtype=torch.float32).to(device)
    out = model(inp)
    #print(out.shape)

    # summary(model, (1, 3, 224, 224))