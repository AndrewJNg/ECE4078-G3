
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
# import slam.aruco_detector as aruco
from slam.robot import Robot
# '''
def estimate_pose(camera_matrix, completed_img_dict):
    camera_offset = -0.35
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    target_dimensions = [
        [0.074, 0.074, 0.095],  # Red Apple
        [0.081, 0.081, 0.0841],  # Green Apple
        [0.075, 0.075, 0.0797],  # Orange
        [0.113, 0.067,  0.0599],  # Mango
        [0.073, 0.073, 0.0957],  # Capsicum
    ]

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target']
        robot_pose = completed_img_dict[target_num]['robot']
        true_height = target_dimensions[target_num - 1][2]

        pixel_height = box[3][0]
        pixel_center = box[0][0]
        # print(pixel_center)
        distance = true_height /  pixel_height * focal_length

        image_width = 640
        x_shift = image_width / 2 - pixel_center
        theta = np.arctan(x_shift / focal_length)
        ang = theta + robot_pose[2][0]

        distance_obj = distance / np.cos(theta)
        x_relative = distance_obj * np.cos(theta)
        y_relative = distance_obj * np.sin(theta)

        target_pose = {'x': 0.0, 'y': 0.0}

        # Modify the following lines to account for the camera offset
        camera_x = camera_offset * np.cos(robot_pose[2][0])
        camera_y = camera_offset * np.sin(robot_pose[2][0])

        target_pose['x'] = (x_relative + camera_x) * np.cos(ang) - (y_relative + camera_y) * np.sin(ang)
        target_pose['y'] = ((x_relative + camera_x) * np.sin(ang) + (y_relative + camera_y) * np.cos(ang)) 
        # print()
        # print(f"x_relative: {x_relative}")
        # print(f"x_camera: {camera_x}")
        # print(f"np.sin(ang): {np.sin(ang)}")

        # print(f"y_relative: {y_relative}")
        # print(f"y_camera: {camera_y}")
        # print(f"np.cos(ang): {np.cos(ang)}")
        # print()
        # print((x_relative + camera_x) * np.sin(ang))
        # print((y_relative + camera_y) * np.cos(ang))
        # print()
        # print(theta*180/np.pi)
        # print(ang*180/np.pi)

        target_pose_dict[target_list[target_num - 1]] = target_pose

    return target_pose_dict
# '''


class aruco_detector:
    def __init__(self, robot, marker_length=0.06):
        self.camera_matrix = robot.camera_matrix
        self.distortion_params = robot.camera_dist

        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, corners,ids,marker_length,camera_matrix,distortion_params):
        # Perform detection
        # corners_1, ids_ori, rejected = cv2.aruco.detectMarkers(
            # img, self.aruco_dict, parameters=self.aruco_params)
        
        # corners = [corner.tolist() for corner in corners]
        # print("corners_1:")
        # print(corners_1)
        # print()
        # print("corners:")
        # print(np.array(corners))

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, distortion_params)
        
        # print()
        # print("ids:")
        # print(ids)
        # print()
        # print("marker length:")
        # print(marker_length)
        # print()
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

if __name__ == "__main__":
    
    detc = Detector("network/scripts/model/yolov8_model_best.pt")
    # img = np.array(Image.open('network/scripts/image_0.png'))
    img = np.array(Image.open('network/scripts/image_2.jpeg'))
    
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



    base_dir = Path('./')
    detector_output, network_vis = detc.detect_single_image(img)
    # print(detector_output)
    
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    aruco_det = aruco_detector(robot, marker_length = 0.06)
    corners =[[[        486,         293],
        [        436,         291],
        [        437,         237],
        [        487,         238.0]]]
    corners = (np.array(corners, dtype=np.float32),)

    ids = np.array([[3]])
    
    landmarks, aruco_img, boundingbox = aruco_det.detect_marker_positions(corners=corners,ids=ids,marker_length = 0.06,camera_matrix = camera_matrix,distortion_params=dist_coeffs)

    # landmarks, aruco_img, boundingbox =  detect_marker_positions(corners, ids = [[3]], marker_length=0.06, camera_matrix=camera_matrix, distortion_params=dist_coeffs)
   
    
#     completed_img_dict ={}
#     for i in range(len(detector_output)):
#         label = detector_output[i][0]
#         box_temp = detector_output[i][1]
        
#         box = [[box_temp[0]],[box_temp[1]],[box_temp[2]],[box_temp[3]]]
#         # robot_coord = np.array([[-0.2   ],[0    ],[np.deg2rad(-90)]])
#         robot_coord = np.array([[0   ],[0    ],[np.deg2rad(0)]])
        
#         completed_img_dict[int(label)] = {'target': np.array(box),
#                                    'robot': robot_coord}
#     # print()
#     # print(np.array(box))
#     x, y, width, height = np.array(box)
#     # corner_points = np.array(box).tolist()
#     # corner_points = np.array(box).reshape((-1, 2))
#     corner_points = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
#     print(corner_points)
#     # target_est = estimate_pose(camera_matrix, completed_img_dict)
#     # detect_marker_positions(self, corners, ids)
#     corner_points = [
#     # Marker 1
#     [
#         [100, 100],
#         [200, 100],
#         [200, 200],
#         [100, 200]
#     ]
#     # Marker 2
#     # [
#     #     [300, 100],
#     #     [400, 100],
#     #     [400, 200],
#     #     [300, 200]
#     # ]
# ]
#     corners = np.array(corner_points, dtype=np.float32)
#     custom_marker_ids = np.array([1], dtype=np.int32)
#     measurements, img_marked, bounding_boxes = detect_marker_positions(corners =  corners, ids=custom_marker_ids,marker_length= 0.095,camera_matrix=camera_matrix,distortion_params=dist_coeffs)


    # print("target_est: ")
    # print(target_est)
    # print("answer: [x = 0.8 , y = 0.2]")
    # imgplot = plt.imshow(network_vis)
    # plt.show()
    imgplot = plt.imshow(aruco_img)
    plt.show()

