
# from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from network.scripts.detector import Detector
from network.scripts.detector import Detector # modified for yolov8
# import TargetPoseEst
from pathlib import Path

def estimate_pose( camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    redapple_dimensions = [0.074, 0.074, 0.087]
    target_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067]
    target_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072]
    target_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
    target_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.088]
    target_dimensions.append(capsicum_dimensions)

    # target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
    target_list = [11, 12, 13, 14, 15]

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        
        camera_offset = 0.035 #3.5cm
        target_pose = {'x': 0.0, 'y': 0.0}
        distance = focal_length * true_height/box[3][0]
         
        target_pose['x'] = robot_pose[0][0] + np.cos(robot_pose[2][0]) * (distance + camera_offset)
        target_pose['y'] = robot_pose[1][0] + np.sin(robot_pose[2][0]) * (distance + camera_offset) 
        
        target_pose_dict[target_list[target_num-1]] = target_pose

        # print(target_pose_dict)
    return target_pose_dict

def fruit_detect(nn_model, camera_matrix, img, robot_pose):
    # nn_model = Detector("network/scripts/model/yolov8_model_best.pt")
    # img = np.array(Image.open('network/scripts/image_0.png'))
    
    # fileK = "{}intrinsic.txt".format('./calibration/param/')
    # camera_matrix = np.loadtxt(fileK, delimiter=',')

    detector_output, network_vis = nn_model.detect_single_image(img)
    # print(len(detector_output))

    completed_img_dict = {}
    for i in range(len(detector_output)):
        label = detector_output[i][0]
        box_temp = detector_output[i][1]
        
        box = [[box_temp[0]],[box_temp[1]],[box_temp[2]],[box_temp[3]]]
        
        completed_img_dict[int(label)] = {'target': np.array(box),
                                   'robot': robot_pose}
    # print()
    # print(completed_img_dict)
    
    target_est = estimate_pose(camera_matrix, completed_img_dict)
    print("target_est: ")
    print(target_est)
    return target_est
    # imgplot = plt.imshow(network_vis)
    # plt.show()