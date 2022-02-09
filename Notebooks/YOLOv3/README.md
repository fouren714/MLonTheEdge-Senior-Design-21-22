# This notebook includes code to train Yolov3-tiny on LPCV.ai data

- Use Google Colab environment to test the code
- To train on any dataset, the first cell must be run before any other

- For the First time, you need to see the code in YOLOv5 notebook to zip the files then you can unzip the files in this code.

In any cell with the command !python detect.py, different versions of the model can be run after they have been created by changing "./runs/train/exp/weights/best.pt" to the correct path to the new model file

- note: As of Dec/21 the export.py was not working to convert the .pt file to tflite file.