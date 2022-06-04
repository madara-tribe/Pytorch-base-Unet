import sys, os
import time
import numpy as np
import torch
import onnxruntime
import cv2

H = W = 256

def inference(onnx_file_name, image_path):
    # Input
    img = cv2.imread(image_path)
    img = cv2.resize(img, (H, W), cv2.INTER_NEAREST)
    img_in = img.transpose(2, 0, 1).reshape(1, 3, H, W)/255
    print("Shape of the network input: ", img_in.shape, img_in.min(), img_in.max())

    # onnx runtime
    ort_session = onnxruntime.InferenceSession(onnx_file_name)
    IMAGE_HEIGHT = ort_session.get_inputs()[0].shape[2]
    IMAGE_WIDTH = ort_session.get_inputs()[0].shape[3]
    input_name = ort_session.get_inputs()[0].name
    print("The model expects input shape: ", ort_session.get_inputs()[0].shape)
    
    # prediction
    print('start calculation')
    start_time = time.time()
    outputs = ort_session.run(None, {input_name: img_in.astype(np.float32)})[0]
    outputs = (outputs[0] * 255).transpose(1, 2, 0).astype(np.float32)
    img_in = (img_in[0] * 255).transpose(1, 2, 0).astype(np.float32)
    save_img = np.hstack([img_in, outputs])
    #print(save_img.shape, img_in.shape, outputs.shape)
    cv2.imwrite('onnx_pred.png', save_img.astype(np.uint8))
    print("Inference Latency (ms) until saved is", (time.time() - start_time)*1000, "[ms]")

if __name__=='__main__':
    image_path = str(sys.argv[1])
    onnx_file_name = str(sys.argv[2])
    inference(onnx_file_name, image_path)


