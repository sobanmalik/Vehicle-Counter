#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time
import cv2
import os


args = {
            "config": './yolo_files/yolov3.cfg',
            "weights": './yolo_files/yolov3.weights',
            "classes": './yolo_files/coco.names.txt'
            
        }

try:
    # load the COCO class labels our YOLO model was trained on
    LABELS = None
    with open(args['classes'], 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    net = cv2.dnn.readNet(args['weights'],args['config'])
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # initialize the video stream, pointer to output video file, and frame dimensions
    print('Testing for:', os.listdir(os.path.join('./testing_video'))[0])
    vs = cv2.VideoCapture('./testing_video/' + os.listdir(os.path.join('./testing_video'))[0])
    (W, H) = (None, None)

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        counter = 0

        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        #if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forwar pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        #initialize our lists of detected bounding boxes, confidences,
        #and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)


        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences,0.5,0.3)
    #     print([LABELS[i] for i in classIDs])
    #     increment counter for every vehicle prediction in a frame
        vehicles = ['car', 'motorbike', 'bicycle', 'bus', 'train', 'aeroplane', 'truck' ]
        for lab in [LABELS[i] for i in classIDs]:
            if lab in vehicles:
                counter += 1
        # ensure at least one detection exists
        if len(idxs) > 0:

            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #           put the counter on the frame
                cv2.putText(frame, 'Total Vehicles in Frame : ' + str(counter), (750, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2,cv2.LINE_AA)

    #         show the video
            cv2.imshow("Image",frame)   
            key = cv2.waitKey(1)
            if key == 27: #esc key stops qthe process
                break


    vs.release()
    
except:
    print('No testing video or yolov3 weights found')

