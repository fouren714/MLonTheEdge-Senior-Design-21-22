# This notebook includes code to train Keras Retinanet on LPCV.ai data

- In order to train over 22 epochs, the patience parameter in train.py must be changed to be equal to that of the epochs, which is 50 by default
- tflite is partially implemented, however there are still errors in its working. After determining intial inference time to be about 3s it seems illogical to continue working

- To train and Test on LPCV data from our Roboflow dataset:
  - Run the first seven cells
    - These cells work to train the model on our dataset, and import all necessary modules for detection
  - Upload the video file "5p5b_03A1.mp4" from the LPCV.ai challenge, or a desired image or video
    - Run the corresponding detect_image or detect_video function
