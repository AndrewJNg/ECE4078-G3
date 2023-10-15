
# from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from network.scripts.detector import Detector
from network.scripts.detector import Detector # modified for yolov8
# import TargetPoseEst
from pathlib import Path
import cv2
import util.measure as measure
import slam.aruco_detector as aruco
from slam.robot import Robot
    
def detect_single_fruit_positions(img,corners,ids,marker_length,camera_matrix,distortion_params):
    # Perform detection
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, distortion_params)
    
    if ids is None:
        return [], img, []

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

    return measurements, img_marked, bounding_boxes

def detect_fruit_landmark(yolov,img,camera_matrix,dist_coeffs):
    target_dimensions = [
            [0.074, 0.074, 0.087],  # Red Apple
            [0.081, 0.081, 0.067],  # Green Apple
            [0.075, 0.075, 0.072],  # Orange
            [0.113, 0.067, 0.058],  # Mango
            [0.073, 0.073, 0.088],  # Capsicum
        ]
    
    detector_output, network_vis = yolov.detect_single_image(img)
    # imgplot = plt.imshow(network_vis)
    # plt.show()
    # print(detector_output)
    
    
    measurements = []
    for i in range(len(detector_output)):
        corners = []
        ids = []
        label = detector_output[i][0]
        # print("label")
        # print(label)
        box_temp = detector_output[i][1]
        
        box = [box_temp[0]],[box_temp[1]],[box_temp[2]],[box_temp[3]]
        corners.append([[
                        [box[0][0]+box[2][0]/2,box[1][0]+box[3][0]/2],
                        [box[0][0]-box[2][0]/2,box[1][0]+box[3][0]/2],
                        [box[0][0]-box[2][0]/2,box[1][0]-box[3][0]/2],
                        [box[0][0]+box[2][0]/2,box[1][0]-box[3][0]/2]
                        ]])
        
        ids.append(int(label)+10)
        corners = (np.array(corners[0], dtype=np.float32),)
        ids = np.array([ids])
        # print()
        # print("before")
        # print(corners)
        # print(ids)
        landmarks, aruco_img, boundingbox = detect_single_fruit_positions(img=img,corners=corners,ids=ids,marker_length = target_dimensions[int(label)-1][0],camera_matrix = camera_matrix,distortion_params=dist_coeffs)
        measurements.append(landmarks[0])
        
        # imgplot = plt.imshow(aruco_img)
        # plt.show()
    # print(corners)
        

    
    return measurements

## testing file
if __name__ == "__main__":
    
    detc = Detector("network/scripts/model/yolov8_model_best.pt")
    # img = np.array(Image.open('network/scripts/image_0.png'))
    # img = np.array(Image.open('network/scripts/image_2.jpeg'))
    img = np.array(Image.open('network/scripts/image_3.png'))
    
    global dist_coeffs
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    
    global camera_matrix
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    
    global scale
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    
    global baseline
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')


    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    aruco_det = aruco.aruco_detector(robot, marker_length = 0.06)
    landmarks_aruco, aruco_img, boundingbox = aruco_det.detect_marker_positions(img)
    print(landmarks_aruco)
    # imgplot = plt.imshow(aruco_img)
    # plt.show()
    # print(detector_output)


    landmarks_fruits = detect_fruit_landmark(yolov=detc,img=img,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs)
    print(landmarks_fruits)
    landmarks_combined = []
    landmarks_combined.extend(landmarks_aruco)
    landmarks_combined.extend(landmarks_fruits)

    # landmarks = np.reshape[(landmarks_aruco,landmarks_fruits),-1]
    print(landmarks_combined)

