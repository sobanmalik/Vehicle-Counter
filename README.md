# Vehicle-Counter
A model which can accurately count the number of cars/vehicles present at a given time in a video.

To test a video, create a folder 'testing_video' and put the video in it. Only one video can be processed at a time.

!! Download yoloV3 weights from https://pjreddie.com/media/files/yolov3.weights and put it in 'yolo_files' folder.

Approach taken:

Tried different Yolo Models on two different use cases: 
1. traffic in the day 
2. traffic in the night

Used YOLO modles pre trained on the COCO dataset. Other models do not work well in detecting grouped objects and are relatively slower.


Got most accurate results with YOLOv3:

- model size: 236MB
- Almost no vehicle is missed in detection.
- Inferencing is a little slow. (Lag is present)
- confidence input = 0.5, threshold = 0.3
