
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
        print(pixel_center)
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
        print()
        print(f"x_relative: {x_relative}")
        print(f"x_camera: {camera_x}")
        print(f"np.sin(ang): {np.sin(ang)}")

        print(f"y_relative: {y_relative}")
        print(f"y_camera: {camera_y}")
        print(f"np.cos(ang): {np.cos(ang)}")
        print()
        print((x_relative + camera_x) * np.sin(ang))
        print((y_relative + camera_y) * np.cos(ang))
        print()
        print(theta*180/np.pi)
        print(ang*180/np.pi)

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
    imgplot = plt.imshow(network_vis)
    plt.show()

