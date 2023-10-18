
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
import json
    
def detect_single_fruit_positions(img,corners,ids,marker_length,camera_matrix,distortion_params):
    # Perform detection
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, distortion_params)
    # print(tvecs)
    
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

def take_marker_pose(box,robot_pose):
    global camera_matrix
    # robot_pose = [0,0,0]
    true_height = 0.06
    focal_length = camera_matrix[0][0]
    camera_offset = 0.110 #3.5cm

    # print(box[0][3])
    
    ## ASSUMING ONLY 1 BOX
    distance = focal_length * true_height/box[0][3]

    x = robot_pose[0] + np.cos(robot_pose[2]) * (distance + camera_offset)
    y = robot_pose[1] + np.sin(robot_pose[2]) * (distance + camera_offset) 
    world_frame_pos = [x,y]

    return world_frame_pos

def detect_fruit_landmark(yolov,img,camera_matrix,dist_coeffs):
    target_dimensions = [
            [0.074, 0.074, 0.135],  # Red Apple X
            [0.081, 0.081, 0.097],  # Green Apple X
            [0.075, 0.075, 0.082],  # Orange 
            [0.113, 0.067, 0.062],  # Mango 
            [0.073, 0.067, 0.120],  # Capsicum X
            '''
            [0.074, 0.074, 0.083],  # Red Apple X
            [0.081, 0.081, 0.067],  # Green Apple X
            [0.075, 0.075, 0.072],  # Orange 
            [0.113, 0.067, 0.058],  # Mango 
            [0.073, 0.067, 0.093],  # Capsicum X            
            '''
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
        box_temp = detector_output[i][1]
        
        x_center = box_temp[0]
        y_center = box_temp[1]
        x_offset = box_temp[2]/2
        y_offset = box_temp[3]/2        

        height, width, channel = img.shape

        wall_tolerance = 30
        min_fruit_bbox = 20
        # print()
        if(np.floor(x_center-x_offset)<=(0+wall_tolerance) or np.ceil(x_center+x_offset)>=(width-wall_tolerance)):
            # print(f"ignore: {label}, due to hitting the sides")
            continue
        elif(np.floor(y_center-y_offset)<=(0+wall_tolerance) or np.ceil(y_center+y_offset)>=(height-wall_tolerance)):
            # print(f"ignore: {label}, due to hitting the ceiling/floor")
            continue
        elif (box_temp[2] <=min_fruit_bbox or box_temp[3] <=min_fruit_bbox):
            # print(f"ignore: {label}, due to being too small")
            continue

        else:
            # continue with finding the landmark
            corners.append([[
                            [x_center+x_offset, y_center+y_offset],
                            [x_center-x_offset, y_center+y_offset],
                            [x_center-x_offset, y_center-y_offset],
                            [x_center+x_offset, y_center-y_offset]
                            ]])
            
            ids.append(int(label)+10)
            corners = (np.array(corners[0], dtype=np.float32),)
            ids = np.array([ids])
            # print(f"Fruit id: {ids[0][0]}, with bbox {box_temp}")

            marker_height = target_dimensions[int(label)-1][2]
            aspect_ratio = box_temp[2]/ box_temp[3] # box_wdith / box_height
            marker_length = aspect_ratio * marker_height
            # marker_length = (target_dimensions[int(label)-1][0] + target_dimensions[int(label)-1][1])/2
            landmarks, fruit_img, boundingbox = detect_single_fruit_positions(img=img,corners=corners,ids=ids,marker_length = marker_length ,camera_matrix = camera_matrix,distortion_params=dist_coeffs)
            measurements.append(landmarks[0])
        

    
    return measurements,network_vis 

## testing file
if __name__ == "__main__":
    
    detc = Detector("network/scripts/model/yolov8_model_best.pt")
    # img = np.array(Image.open('network/scripts/image_0.png'))
    # img = np.array(Image.open('network/scripts/image_2.jpeg'))
    img = np.array(Image.open('network/scripts/image_3.png'))
    # img = np.array(Image.open('network/scripts/image_4.png'))
    # img = np.array(Image.open('network/scripts/image_5.png'))
    # img = np.array(Image.open('network/scripts/image_6.png'))
    
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
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    landmarks_aruco, aruco_img, boundingbox, aruco_id = aruco_det.detect_marker_positions(img)

    
    aruco_base_dict = {}
    marker_pose = None
    robot_pose = [0,0,90*np.pi/180]
    boundingbox = np.array(boundingbox)

    if len(aruco_id) ==1:
        print(boundingbox)
        x,y= take_marker_pose(boundingbox,robot_pose)
        aruco_base_dict[f'aruco{int(aruco_id[0][0])}_0'] ={'x': x,'y': y}
    elif len(aruco_id) >=2:
        print(boundingbox)
        
        x_values = np.array([item[0] for item in boundingbox])
        print(f"aruco_id: {aruco_id} bbox: {boundingbox}, x: {x_values}")
        offset = abs(320*np.ones_like(x_values) - x_values)
        print(f"offset: {offset}")
        index = np.argmin(offset)
        print(f"min: {index}")
        id = aruco_id[index]
        print(f"id: {id}")
        
        x,y= take_marker_pose([boundingbox[index]],robot_pose)
        aruco_base_dict[f'aruco{int(id[0])}_0'] ={'x': x,'y': y}
        
        
    with open('lab_output/base_map.txt', 'w') as f:
        json.dump(aruco_base_dict, f, indent=4)

    imgplot = plt.imshow(aruco_img)
    plt.show()
    # print(aruco_base_dict)

    print(f"aruco_landmarl: {landmarks_aruco}")
    # imgplot = plt.imshow(aruco_img)
    # plt.show()
    # print(detector_output)

    landmarks_fruits,network_img = detect_fruit_landmark(yolov=detc,img=img,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs)
    print(f"fruits_landmarl: {landmarks_fruits}")
    
    landmarks_combined = []
    landmarks_combined.extend(landmarks_aruco)
    landmarks_combined.extend(landmarks_fruits)
    print(landmarks_combined)
    
    # imgplot = plt.imshow(network_img)
    # plt.show()

