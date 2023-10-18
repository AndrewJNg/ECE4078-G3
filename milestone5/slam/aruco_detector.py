# detect ARUCO markers and estimate their positions
import numpy as np
import cv2
import os, sys

sys.path.insert(0, "{}/util".format(os.getcwd()))
import util.measure as measure

class aruco_detector:
    def __init__(self, robot, marker_length=0.06):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, img):
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # print("corners:")
        # print(corners)
        # print()
        # print("ids:")
        # print(ids)
        # print()
        # print("marker length:")
        # print(self.marker_length)
        # print()
        if ids is None:
            return [], img, [],[]

        # Compute the marker positions
        measurements = []
        bounding_boxes = []  # Store bounding boxes here
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i, 0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids:
                continue
            else:
                seen_ids.append(idi)

            lm_tvecs = tvecs[ids == idi].T
            lm_bff2d = np.block([[lm_tvecs[2, :]], [-lm_tvecs[0, :]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1, 1)

            # Calculate the bounding box from the detected corners
            bounding_box = cv2.boundingRect(corners[i][0])  # Get bounding box for the i-th marker

            # Create a Marker object with measurement and bounding box
            lm_measurement = measure.Marker(lm_bff2d, idi)
            measurements.append(lm_measurement)
            bounding_boxes.append(bounding_box)

        # Draw markers and bounding boxes on the image copy
        img_marked = img.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            cv2.rectangle(img_marked, (x, y), (x + w, y + h), (0, 255, 0), 2)

        
        return measurements, img_marked, bounding_boxes,ids
