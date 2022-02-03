model_path = '../best-int8_edgetpu.tflite'
interpreter = tflite.Interpreter(model_path)
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])