import tflite_runtime.interpreter as tflite
from PIL import Image
import datetime
import numpy as np
import os
import cv2


def process_image(img):
    image = np.array(img.resize((320, 320)), dtype="int8")
    #     image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def detect_image(image, interpreter):
    pimage = process_image(image)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, np.array(pimage, dtype="uint8"))
    start = datetime.datetime.now()
    interpreter.invoke()
    time = datetime.datetime.now() - start
    outs = interpreter.get_tensor(output_index)
    shape = np.array(pimage).shape
    outs = [np.array(outs)]
    image = np.array(image)
    return outs, time


def detect_video(video, interpreter):
    # video_path = os.path.join(video)
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    count = 0
    time_array = []
    while success:
        outs, time = detect_image(Image.fromarray(image), interpreter)
        time_array.append(time)
        success, image = vidcap.read()
        count += 1
    print(np.mean(np.array(time_array)))
    return


model_path = "test.tflite"
interpreter = tflite.Interpreter(model_path)
interpreter = tflite.Interpreter(
    model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
)
# import tensorflow as tf
# interpreter = tf.lite.Interpreter(model_path)
print(type(interpreter))

interpreter.allocate_tensors()
image = "test.jpg"
image = Image.open(image)
detect_image(image, interpreter)
video = "4p1b_01A2.m4v"
detect_video(video, interpreter)
