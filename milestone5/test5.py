
# from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from network.scripts.detector import Detector
from network.scripts.detector import Detector # modified for yolov8
# import TargetPoseEst
from pathlib import Path

# '''
def estimate_pose(camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    target_dimensions = [
        [0.074, 0.074, 0.087],  # Red Apple
        [0.081, 0.081, 0.067],  # Green Apple
        [0.075, 0.075, 0.072],  # Orange
        [0.113, 0.067, 0.058],  # Mango
        [0.073, 0.073, 0.088]   # Capsicum
    ]

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target']  # [[x], [y], [width], [height]]
        robot_pose = completed_img_dict[target_num]['robot']  # [[x], [y], [theta]]
        true_height = target_dimensions[target_num - 1][2]
        
        # Extract relevant information
        x, y, width, height = box[0, 0], box[1, 0], box[2, 0], box[3, 0]
        robot_x, robot_y, robot_theta = robot_pose[0, 0], robot_pose[1, 0], robot_pose[2, 0]

        # Calculate the distance from the camera to the object using the focal length
        distance = (true_height * focal_length) / height

        # Calculate the object's 2D position in the camera frame
        object_x = (x - camera_matrix[0, 2]) * distance / camera_matrix[0, 0]
        object_y = (y - camera_matrix[1, 2]) * distance / camera_matrix[1, 1]

        # Calculate the object's 3D position in the world frame
        world_x = robot_x + object_x * np.cos(robot_theta) - object_y * np.sin(robot_theta)
        world_y = robot_y + object_x * np.sin(robot_theta) + object_y * np.cos(robot_theta)

        # The object's orientation is assumed to be aligned with the robot's orientation
        object_theta = robot_theta

        # Store the estimated pose
        # target_pose_dict[target_num] = {
        #     'position': [world_x, world_y],
        #     'orientation': object_theta
        # }
        
        target_pose = {'x': 0.0, 'y': 0.0}
        target_pose['x'] = world_x
        target_pose['y'] = world_y
        
        target_pose_dict[target_list[target_num - 1]] = target_pose

    return target_pose_dict
# '''
'''
def estimate_pose(camera_matrix, completed_img_dict):
    camera_offset = -0.035
    focal_length = camera_matrix[0][0]
    target_dimensions = [
        [0.074, 0.074, 0.087],  # Red Apple
        [0.081, 0.081, 0.067],  # Green Apple
        [0.075, 0.075, 0.072],  # Orange
        [0.113, 0.067, 0.058],  # Mango
        [0.073, 0.073, 0.088],  # Capsicum
    ]

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        # This is the default code which estimates every pose to be (0,0)
        target_pose = {'x': 0.0, 'y': 0.0}
        d = focal_length * true_height/box[3][0]
        u_0 = camera_matrix[0][2]
        theta_f = np.arctan((box[0][0] - u_0)/focal_length)
        target_pose['x'] = robot_pose[0][0] + (d + camera_offset)*np.cos(robot_pose[2][0] + theta_f)
        target_pose['y'] = robot_pose[1][0] + (d + camera_offset)*np.sin(robot_pose[2][0] + theta_f)
        
        target_pose_dict[target_list[target_num-1]] = target_pose

    return target_pose_dict
'''

if __name__ == "__main__":
    
    # detc = Detector("network/scripts/model/yolov8_model_best.pt")
    # # img = np.array(Image.open('network/scripts/image_0.png'))
    # img = np.array(Image.open('network/scripts/image_2.jpeg'))
    
    # fileK = "{}intrinsic.txt".format('./calibration/param/')
    # camera_matrix = np.loadtxt(fileK, delimiter=',')
    # base_dir = Path('./')

    camera_matrix = [[1.07453000e+03, 0.00000000e+00, 2.74690405e+02],
                    [0.00000000e+00, 1.07258648e+03, 1.94508578e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    
    # detector_output, network_vis = detc.detect_single_image(img)
    detector_output = [['1', np.array([     120-70/2,      290-80/2,       70,       80])]]
    # print(detector_output)

    completed_img_dict ={}
    for i in range(len(detector_output)):
        label = detector_output[i][0]
        box_temp = detector_output[i][1]
        
        box = [[box_temp[0]],[box_temp[1]],[box_temp[2]],[box_temp[3]]]
        # robot_coord = np.array([[-0.2   ],[0    ],[np.deg2rad(-90)]])
        robot_coord = np.array([[0   ],[0    ],[np.deg2rad(0)]])
        
        completed_img_dict[int(label)] = {'target': np.array(box),
                                   'robot': robot_coord}
    # print()
    # print(completed_img_dict)
    
    target_est = estimate_pose(camera_matrix, completed_img_dict)
    print("target_est: ")
    print(target_est)
    print("answer: [x = 0.8 , y = 0.2]")
    # imgplot = plt.imshow(network_vis)
    # plt.show()

