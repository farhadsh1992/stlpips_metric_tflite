

import tensorflow as tf


def im2tensor(path:str="", image:str=""):
    if path != ""
      image = tf.io.read_file(path)
      image = tf.io.decode_jpeg(image)
      
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (416 , 416))
    image = tf.expand_dims(image, axis=0)
    image = tf.transpose(image, (0,3,1,2))
    # image =  tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,3,1,2)))(image[0])
    return image
  
  
class stlpips_metric(tf.keras.layers.Layer):
    def __init__(self, tflite_model_path: str="./ShiftTolerant_LPIPS_TFLite/ShiftTolerant_LPIPS_TFLite.tflite"):
        super(stlpips_metric, self).__init__()
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # get inputs and output shapes
        input_shape1 = self.input_details[0]['shape']
        input_shape2 = self.input_details[1]['shape']
        output_shape = self.output_details[0]['shape']
        
    def call(self, img0: tf.Tensor, img1: tf.Tensor)-> float:
        self.interpreter.set_tensor(self.input_details[0]['index'], img0)
        self.interpreter.set_tensor(self.input_details[1]['index'], img1)

    
        self.interpreter.invoke()

        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0][0][0]