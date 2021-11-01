# This notebook includes code to train Yolov5s on LPCV.ai data as well as the Kitti Dataset

- It is best to run this notebook in a Google Colab environment
  - If you run on a personal computer, the entire Kitti dataset will be downloaded to your computer
- To train on any dataset, the first cell must be run before any other

- To train on LPCV data from our Roboflow dataset:

  - Rename the folder created by the initial cell from "LPCV_data-2" to "LPCV_data_2"
  - Inside that folder, in the file data.yaml, change all "LPCV_data-2" to "LPCV_data_2"
  - Upload the video file "5p5b_03A1.mp4" from the LPCV.ai challenge
  - Scroll to last section of the notebook, after the cell titled "END OF LPCV DATA FROM WEBSITE"
  - Run the cells after
    - The first cell trains Yolov5 on our data
    - The second cell detects objects in the "5p5b_03A1.mp4" video file
    - The third cell creates keras, tensorflow, and tflite versions of the model

- To train on the Kitti dataset

  - Upload the kittidata.yaml file to the yolov5 folder in your Colab environment
  - Run all cells until the cell titled "END OF KITTI DATASET"
    - All but the final two cells involve fetching and modifying the Kitti data for Yolov5 use
    - The second to last cell trains Yolov5 on the dataset
    - The final cell can be used to test an image titled "test.png"
      - This image should be uploaded to the yolov5 folder
      - "./runs/train/exp/weights/best.pt" may need to be changed based on where the training weights are saved by adding the correct number to the end of "exp"

- To train on the LPCV.ai data from the challenges files:
  - This is not working correcrtly at this point and should be ignored until noted


In any cell with the command !python detect.py, different versions of the model can be run after they have been created by changing "./runs/train/exp/weights/best.pt" to the correct path to the new model file
