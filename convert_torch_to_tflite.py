



import torch
import stlpips
import util as utils
import torch
import numpy as np
import cv2
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def main():
  # iimg = torch.ones_like((4, 3,416,416), dtype=torch.float32)
  batch_number = 5
  img0 = torch.randn(batch_number, 3,416,416)
  img1 = torch.randn(batch_number, 3,416,416)

  onnx_out_path = 'cache/out_batch{}.onnx'.format(batch_number)
  tf_model_path = "cache/tfmodel_batch{}".format(batch_number)
  tflite_model_path = "./ShiftTolerant_LPIPS_TFLite/ShiftTolerant_LPIPS_TFLite_batch{}.tflite".format(batch_number)
  
  
  stlpips_metric = stlpips.LPIPS(net="alex", variant="shift_tolerant")
  torch.onnx.export(stlpips_metric, (img0,img1) , onnx_out_path)
  
  # Load the ONNX model
  model = onnx.load(onnx_out_path)

  # Check that the IR is well formed
  onnx.checker.check_model(model)

  # Print a Human readable representation of the graph
  re = onnx.helper.printable_graph(model.graph)
  onnx_model = onnx.load(onnx_out_path)
  
  

  tf_rep = prepare(onnx_model)
  model = tf.saved_model.load(tf_model_path)
  model.trainable = False

  input_tensor1 = tf.random.uniform([batch_size, channels, height, width])
  converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
  tflite_model = converter.convert()



  # Save the model
  with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
    
    
if __name__ == "__main__":
  main():