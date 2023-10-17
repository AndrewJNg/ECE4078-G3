
# from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from network.scripts.detector import Detector
from network.scripts.detector import Detector # modified for yolov8
# import TargetPoseEst
from pathlib import Path

def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = [
        [0.074, 0.074, 0.087],  # Red Apple
        [0.081, 0.081, 0.067],  # Green Apple
        [0.075, 0.075, 0.072],  # Orange
        [0.113, 0.067, 0.058],  # Mango
        [0.073, 0.073, 0.088],  # Capsicum
    ]

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target']
        robot_pose = completed_img_dict[target_num]['robot']
        true_height = target_dimensions[target_num - 1][2]

        # Compute the pose of the target based on bounding box info and robot's pose
        pixel_height = box[3][0]  # Use box[3][0] to access height
        pixel_center = box[0][0] + box[2][0] / 2  # Access x and width from box
        distance = true_height / pixel_height * focal_length

        image_width = 640
        x_shift = image_width / 2 - pixel_center
        theta = np.arctan(x_shift / focal_length)
        ang = theta + robot_pose[2][0]

        distance_obj = distance / np.cos(theta)
        x_relative = distance_obj * np.cos(theta)
        y_relative = distance_obj * np.sin(theta)

        target_pose = {'x': 0.0, 'y': 0.0}

        target_pose['x'] = x_relative * np.cos(ang) - y_relative * np.sin(ang)
        target_pose['y'] = x_relative * np.sin(ang) + y_relative * np.cos(ang)

        target_pose_dict[target_list[target_num - 1]] = target_pose










    return target_pose_dict


if __name__ == "__main__":
    
    detc = Detector("network/scripts/model/yolov8_model_best.pt")
    # img = np.array(Image.open('network/scripts/image_0.png'))
    img = np.array(Image.open('network/scripts/image_2.jpeg'))
    
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')

    detector_output, network_vis = detc.detect_single_image(img)
    # print(detector_output)

    completed_img_dict ={}
    for i in range(len(detector_output)):
        label = detector_output[i][0]
        box_temp = detector_output[i][1]
        
        box = [[box_temp[0]],[box_temp[1]],[box_temp[2]],[box_temp[3]]]
        # robot_coord = np.array([[-0.2   ],[0.4    ],[np.deg2rad(-90)]])
        robot_coord = np.array([[0   ],[0    ],[np.deg2rad(0)]])
        
        completed_img_dict[int(label)] = {'target': np.array(box),
                                   'robot': robot_coord}
    # print()
    # print(completed_img_dict)
    
    target_est = estimate_pose(base_dir, camera_matrix, completed_img_dict)
    print("target_est: ")
    print(target_est)
    print("answer: [x = 0.8 , y = 0.2]")
    imgplot = plt.imshow(network_vis)
    plt.show()

