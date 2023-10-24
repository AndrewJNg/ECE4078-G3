# Import modules
import pygame
import math
# from queue import PriorityQueue
import json
import ast
import numpy as np


def getPath(dist_to_fruit, rorbot_pose, fruits_arr, fruit_order):
    """
    Returns: 
        - `2D arr` with 3 coordinates
        - `float` for heading of robot

    Params:
        - `tolerance` distance when robot will take picture from fruit
        - `start_pos`
        - `fruits_arr` fruits positions
        - `shortestPath` from getShortestPath function
    """
    # Psuedo Code:
    # Given (x1,y1) (x2,y2) (x3,y3)
    # Find all possible location to take fruit
    # Choose shortest
    final_visit_pos = []
    # min_dir_arr  = [] # TODO comment
    for index, currentFruit in enumerate(fruits_arr):
        # print(f'\n{fruit_order[index]}') # Debug
        # Get current pos
        x_i, y_i = rorbot_pose[0], rorbot_pose[1]

        # Get fruit position
        x0, y0 = currentFruit

        # Generate locations for non-diagonal visit_pos
        visit_pos_arr = {}
        visit_pos_arr['right'] =    [round(x0+dist_to_fruit,1), y0]
        visit_pos_arr['up'] =       [x0, round(y0+dist_to_fruit,1)]
        visit_pos_arr['left'] =     [round(x0-dist_to_fruit,1), y0]
        visit_pos_arr['down'] =     [x0, round(y0-dist_to_fruit,1)]
        # Generate locations for diagonal visit_pos
        diag_visit_pos_arr = {}
        diag_visit_pos_arr['upleft'] =       [round(x0-dist_to_fruit,1), round(y0+dist_to_fruit,1)]
        diag_visit_pos_arr['upright'] =      [round(x0+dist_to_fruit,1), round(y0+dist_to_fruit,1)]
        diag_visit_pos_arr['downleft'] =     [round(x0-dist_to_fruit,1), round(y0-dist_to_fruit,1)]
        diag_visit_pos_arr['downright'] =    [round(x0+dist_to_fruit,1), round(y0-dist_to_fruit,1)]
        # print('\nFruit {}: {}'.format(index, visit_pos_arr))  # Debug
        
        unmerged_final_visit_pos = []
        # Appending non colliding positions, while adding distance as second index
        # if len(visit_pos_arr) != 0: # Prioritize non-diagonal visit_pos
        for dir, pos in visit_pos_arr.items():
            x_f, y_f = pos[0], pos[1]
            dist = np.hypot(x_f-x_i, y_f-y_i)
            unmerged_final_visit_pos.append([pos, dist])

        # else: # Take diagonal
        for dir, pos in diag_visit_pos_arr.items():
            x_f, y_f = pos[0], pos[1]
            dist = np.hypot(x_f-x_i, y_f-y_i)
            unmerged_final_visit_pos.append([pos, dist])
        
        # Merge with diagonal visit_pos
        final_visit_pos.append(unmerged_final_visit_pos)
    return final_visit_pos

# Based on search order, generate fruit_arr and obstacles_arr containing fruit position
def getFruitArr(search_list, fruit_list, fruit_true_pos):
    search_fruits = []
    # markerNum = len(obstacles_arr)
    # search_fruits = [[0] * 2 for i in range(len(search_list))]
    for i, to_search in enumerate(search_list):
        for j, fruit in enumerate(fruit_list):
            if to_search == fruit:
                if len(search_fruits) == 0:
                    search_fruits = np.array([fruit_true_pos[j]])
                else:
                    search_fruits = np.append(search_fruits, [fruit_true_pos[j]], axis=0)

    return search_fruits

def round_to_accuracy(number, accuracy=0.4):
    if accuracy <= 0:
        raise ValueError("Accuracy must be greater than 0")
    
    rounded_value = round(number / accuracy) * accuracy
    return rounded_value

def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = round_to_accuracy(gt_dict[key]['x'])
            y = round_to_accuracy(gt_dict[key]['y'])

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos

def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

# For debugging
def getFromFile(fname):
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(fname)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    return search_list, fruits_list, fruits_true_pos, aruco_true_pos

# Main function
def generateWaypoints(robot_pose = [0.,0.,0.], search_list={}, fruits_list={}, fruits_true_pos={}):
    # Define params:
    dist_to_fruit = 0.25     # distance when robot will take picture from fruit

    debug = 0
    # For debugging, to run independently
    if debug and search_list == {}:
        groundtruth = 1
        if groundtruth == 1:
            fname = 'M4_true_map.txt'
            print('Reading from ground truth map')
        else:
            fname = 'M5_est_map.txt'
            print('Reading from estimate SLAM map')
        search_list, fruits_list, fruits_true_pos, _ = getFromFile(fname)
    
    # Extracting data from param
    search_fruits = getFruitArr(search_list, fruits_list, fruits_true_pos)
    waypoints = getPath(dist_to_fruit, robot_pose, search_fruits, search_list)
    
    # Debug
    # print(f'########## Debug ##########\nsearch_list:\n{search_list}\n\nfruits_list:\n{fruits_list}\n\nfruits_true_pos:\n{fruits_true_pos}\n\naruco_true_pos:\n{aruco_true_pos}\n\n')
    # print(f'search_fruits:\n{search_fruits}') 
    # print(f'obstacles_arr:\n{obstacles_arr}')

    return waypoints

# Debug
# waypoints = generateWaypoints(log = 1)