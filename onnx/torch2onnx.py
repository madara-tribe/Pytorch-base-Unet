import sys
sys.path.append('../')
import numpy as np
import torch
from cfg import Cfg
#from unet import UNet
from unet.custom_unet import UNet

H = W = 256
def convert_to_onnx(config, weight_path, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = UNet(inc=3, num_cls=config.classes) 
    model = UNet(in_channel=3, out_channel=config.classes)
    model.to(device)
    model.load_state_dict(torch.load(weight_path))
    print('Loading weights from %s... Done!' % (weight_path))
    
    input_layer_names = ["input"]
    output_layer_names = ["output"]
    x = torch.randn(1, 3, H, W).to(device)
    onnx_file_name = "unet{}_{}.onnx".format(H, W)
    
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      verbose=True,
                      opset_version=11,
                      input_names=input_layer_names,
                      output_names=output_layer_names)
    print('Onnx model exporting done as filename {}'.format(onnx_file_name))
    
if __name__ == '__main__':
    weightfile = '../checkpoints/checkpoint_epoch30.pth'
    batch_size = 1
    config = Cfg
    convert_to_onnx(config, weightfile, batch_size)


