# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5=capsicum, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
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

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose

        camera_offset = 0.0 #2.0cm
        target_pose = {'x': 0.0, 'y': 0.0}
        distance = focal_length * true_height/box[3][0]
         
        target_pose['x'] = robot_pose[0][0] + np.cos(robot_pose[2][0]) * (distance + camera_offset)
        target_pose['y'] = robot_pose[1][0] + np.sin(robot_pose[2][0]) * (distance + camera_offset) 
        
        target_pose_dict[target_list[target_num-1]] = target_pose

        ######################
        # x = box[0]*np.cos(robot_pose[2]) - box[1]*np.sin(robot_pose[2]) + robot_pose[0]
        # y = box[0]*np.sin(robot_pose[2]) + box[1]*np.cos(robot_pose[2]) + robot_pose[1]
        # target_pose = {'x': x, 'y': y}
        
        # target_pose_dict[target_list[target_num-1]] = target_pose

        ###########################################
        # print(target_pose_dict)
    return target_pose_dict

# merge the estimations of the targets so that there are at most 1 estimate for each target type
def merge_estimations(target_map):
    target_map = target_map
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    target_est = {}
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('redapple'):
                redapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each target type.
    # Replace it with a better merge solution.

    # check to see if the count is more than 1, if it is, get the average coordinate  
    ###############################################################################################
    #  original
    # if len(redapple_est) > num_per_target:
    #     redapple_est = ([np.sum(redapple_est,axis=0)/len(redapple_est)])
    #     mean = np.mean(redapple_est)

    # if len(greenapple_est) > num_per_target:
    #     greenapple_est = ([np.sum(greenapple_est,axis=0)/len(greenapple_est)])

    # if len(orange_est) > num_per_target:
    #     orange_est = ([np.sum(orange_est,axis=0)/len(orange_est)])

    # if len(mango_est) > num_per_target:
    #     mango_est = ([np.sum(mango_est,axis=0)/len(mango_est)])

    # if len(capsicum_est) > num_per_target:
    #     capsicum_est = ([np.sum(capsicum_est,axis=0)/len(capsicum_est)])
    ###############################################################################################
    #  mean
    # if len(redapple_est) > num_per_target:
    #     redapple_est = np.mean(redapple_est)

    # if len(greenapple_est) > num_per_target:
    #     greenapple_est = np.mean(greenapple_est)

    # if len(orange_est) > num_per_target:
    #     orange_est = np.mean(orange_est)

    # if len(mango_est) > num_per_target:
    #     mango_est = np.mean(mango_est)

    # if len(capsicum_est) > num_per_target:
    #     capsicum_est = np.mean(capsicum_est)
    ###############################################################################################
    #  Interquartile range (IQR)
    # https://www.askpython.com/python/examples/how-to-determine-outliers#:~:text=If%20the%20value%2Fdata%20point,will%20be%20considered%20an%20outlier.&text=We%20import%20numpy%20module%20for,threshold%20to%201.5%20times%20iqr%20.
    if len(redapple_est) > num_per_target:
        data = redapple_est

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1   
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = np.where((data < lower_fence) | (data > upper_fence))   # rejected values
        valid_val = np.where((data >= lower_fence) | (data <= upper_fence))

        redapple_est = np.mean(valid_val) # only get the average of valid values

    if len(greenapple_est) > num_per_target:
        data = greenapple_est

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1   
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = np.where((data < lower_fence) | (data > upper_fence))   # rejected values
        valid_val = np.where((data >= lower_fence) | (data <= upper_fence))

        greenapple_est = np.mean(valid_val) # only get the average of valid values

    if len(orange_est) > num_per_target:
        data = orange_est

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1   
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = np.where((data < lower_fence) | (data > upper_fence))   # rejected values
        valid_val = np.where((data >= lower_fence) | (data <= upper_fence))

        orange_est = np.mean(valid_val) # only get the average of valid values

    if len(mango_est) > num_per_target:
        data = mango_est

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1   
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = np.where((data < lower_fence) | (data > upper_fence))   # rejected values
        valid_val = np.where((data >= lower_fence) | (data <= upper_fence))

        mango_est = np.mean(valid_val) # only get the average of valid values

    if len(capsicum_est) > num_per_target:
        data = capsicum_est

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1   
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = np.where((data < lower_fence) | (data > upper_fence))   # rejected values
        valid_val = np.where((data >= lower_fence) | (data <= upper_fence))

        capsicum_est = np.mean(valid_val) # only get the average of valid values
    ###############################################################################################

    



    for i in range(num_per_target):
        try:
            target_est['redapple_'+str(i)] = {'x':redapple_est[i][0], 'y':redapple_est[i][1]}
        except:
            pass
        try:
            target_est['greenapple_'+str(i)] = {'x':greenapple_est[i][0], 'y':greenapple_est[i][1]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'x':orange_est[i][0], 'y':orange_est[i][1]}
        except:
            pass
        try:
            target_est['mango_'+str(i)] = {'x':mango_est[i][0], 'y':mango_est[i][1]}
        except:
            pass
        try:
            target_est['capsicum_'+str(i)] = {'x':capsicum_est[i][0], 'y':capsicum_est[i][1]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are only one estimate for each target type
    target_est = merge_estimations(target_map)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')