import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


# Using detector class to match previous resnet version
class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.class_colour = {
            # class: (B,G,R) 
            '0': (220, 220, 220),   #'background'
            '1': (0, 0, 255),       #'red apple'
            '2': (0, 255, 0),       #'green apple'
            '3': (0, 128, 255),     #'orange'
            '4': (0, 255, 255),     #'mango'
            '5': (0, 128, 0),       #'capsicum'
        }

    def detect_single_image(self, img):
        """
        function input:
            img: image file given by opencv2 - cv2.imread() function
        
        function output:
            boundary_boxes: list of lists, box info [label,[x,y,width,height]] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        boundary_boxes = self._get_bounding_boxes(img)
        img_out = deepcopy(img)
        # boundary_boxes = [['1', np.array([     120-70/2,      290-80/2,       70,       80])]]

        # draw bounding boxes on the image
        for box in boundary_boxes:
            
            #  get bounding box to integer values (rounded values to draw on pixel)
            xyxy = ops.xywh2xyxy(box[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            print(x1)
            print(y1)
            print(x2)
            print(y2)
            print()

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[box[0]], thickness=3)

            # draw class label
            img_out = cv2.putText(img_out, box[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                self.class_colour[box[0]], 2)
        # print(boundary_boxes)
        return boundary_boxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        input:
            cv_img: image file given by opencv2 - cv2.imread() function
            model_path: trained YOLOv8 model
        output:
            bounding_boxes: return bounding box values, with format [label, [x,y,width,height] ] 
        """

        # predict using yolov
        predictions = self.model.predict(cv_img, imgsz=640, verbose=False)

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                # bounding format in [x, y, width, height]
                box_cord = box.xywh[0]
                box_label = box.cls  # class label of the box
                bounding_boxes.append([prediction.names[int(box_label)], np.asarray(box_cord.cpu())])

        return bounding_boxes


# Test script
if __name__ == '__main__':
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # load best model
    yolo = Detector(f'{script_dir}/model/yolov8_model_best.pt')

    # load test image
    img = cv2.imread(f'{script_dir}\image_0.png')

    # get prediction
    boundary_boxes, img_out = yolo.detect_single_image(img)

    cv2.imshow('Predict', img_out)
    cv2.waitKey(0)