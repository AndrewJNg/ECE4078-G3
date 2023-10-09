# Optional TODO: 
# Modify getFruitArr as the current code only works if there is 5 fruits' location input into the ground truth.


# Import modules
import pygame
import math
# from queue import PriorityQueue
import json
import ast
import numpy as np


def getPath(tolerance, start_pos, fruits_arr, obstacles_arr, fruit_order):
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
    min_dir_arr  = [] # TODO comment
    for index, currentFruit in enumerate(fruits_arr):
        # Get start position
        if index-1 >= 0:
            x_i, y_i = fruits_arr[index-1][0], fruits_arr[index-1][1]
        else:
            x_i, y_i = start_pos[0], start_pos[1]

        # Get fruit position
        x0, y0 = currentFruit

        # Generate locations
        visit_pos_arr = {}
        visit_pos_arr['right'] =    [round(x0+tolerance,1), y0, round(x0+0.4,1), y0]
        visit_pos_arr['up'] =       [x0, round(y0+tolerance,1), x0, round(y0+0.4,1)]
        visit_pos_arr['left'] =     [round(x0-tolerance,1), y0, round(x0-0.4,1), y0]
        visit_pos_arr['down'] =     [x0, round(y0-tolerance,1), x0, round(y0-0.4,1)]
        visit_pos_arr['upright'] =      [round(x0+tolerance,1), round(y0+tolerance,1), round(x0+0.4,1), round(y0+0.4,1)]
        visit_pos_arr['upleft'] =       [round(x0-tolerance,1), round(y0+tolerance,1), round(x0-0.4,1), round(y0+0.4,1)]
        visit_pos_arr['downleft'] =     [round(x0-tolerance,1), round(y0-tolerance,1), round(x0-0.4,1), round(y0-0.4,1)]
        visit_pos_arr['downright'] =    [round(x0+tolerance,1), round(y0-tolerance,1), round(x0+0.4,1), round(y0-0.4,1)]
        # print('\nFruit {}: {}'.format(index, visit_pos_arr))  # Debug
        # Create a list to store positions to remove
        positions_to_remove = []
        # Check if obstacle is in each location
        # print("") # Debug
        for obstacle_idx, obstacle_pos in enumerate(obstacles_arr):
            x_o, y_o = obstacle_pos[0], obstacle_pos[1]
            for dir, pos in visit_pos_arr.items():
                if (x_o == pos[2] and y_o == pos[3]) or (abs(pos[2]) == 1.6) or (abs(pos[3]) == 1.6):
                    # print('{} visit_pos clashes with obstacle {} at {} [{},{}]'.format(fruit_order[index], obstacle_idx, dir, pos[2],pos[3]))    # Debug
                    positions_to_remove.append(dir)
        print(visit_pos_arr)
        print(visit_pos_arr['down'])
        # Remove the positions that clashed with obstacles
        for dir in positions_to_remove:
            try: 
                visit_pos_arr.pop(dir)
            except:
                continue
        # print(visit_pos_arr) # Debug
        # Calculate distance for each location
        dist_dic = {}
        min_dist = 10000
        for dir, pos in visit_pos_arr.items():
            x_f, y_f = pos[0], pos[1]
            dist = np.hypot(x_f-x_i, y_f-y_i)
            dist_dic[dir] = dist

            # Update min_dist
            if dist < min_dist:
                min_dir = dir
        
        # visit_pos for current fruit
        x_f, y_f = visit_pos_arr[min_dir][0], visit_pos_arr[min_dir][1]

        heading = math.atan2(y_f-y_i, x_f-x_i)

        # Choose optimal visit_pos
        final_visit_pos.append([x_f, y_f, heading])

        # min_dir_arr.append(min_dir) # TODO comment
        
    # print('\nWhere robot is relative to fruit: {}\n'.format(min_dir_arr))   # Debug
    return final_visit_pos

# Based on search order, generate fruit_arr containing fruit position
"""def getFruitArr(all_fruits, search_list_file_name):
    search_fruits = all_fruits
    # Open the file for reading
    with open(search_list_file_name, 'r') as file:
        # Read the contents of the file into a list
        lines = file.readlines()

    # Strip any leading or trailing whitespace from each line and store them in a list
    data = [line.strip() for line in lines]

    for i in data:
        if data == 'redapple':
            search_fruits.append(all_fruits[0])
        if data == 'greenapple':
            search_fruits.append(all_fruits[1])
        if data == 'orange':
            search_fruits.append(all_fruits[2])
        if data == 'mango':
            search_fruits.append(all_fruits[3])
        if data == 'capsicum':
            search_fruits.append(all_fruits[4])
    print(data)
    print(search_fruits)
    return search_fruits"""

# Based on search order, generate fruit_arr containing fruit position
def getFruitArr(all_fruits, search_list):
    search_fruits = []

    for data in search_list:
        if data == 'redapple':
            search_fruits.append(all_fruits[0])
        if data == 'greenapple':
            search_fruits.append(all_fruits[1])
        if data == 'orange':
            search_fruits.append(all_fruits[2])
        if data == 'mango':
            search_fruits.append(all_fruits[3])
        if data == 'capsicum':
            search_fruits.append(all_fruits[4])

    # print(search_fruits) # Debug
    return search_fruits

# For running this file independently
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


# Main function
def generateWaypoints(search_list):
    # Define params:
    start_pos = [0,0]   # start pos
    tolerance = 0.2     # distance when robot will take picture from fruit

    all_fruits = [[0] * 2 for i in range(5)]
    obstacles_arr = [[0] * 2 for i in range(10)]

    # Extracting data from ground truth file
    ground_truth_fname = 'M4_true_map.txt'
    f = open(ground_truth_fname, "r")
    data = json.loads(f.read())
    count = 0

    # for i in data:
    #     if count > 9:
    #         if count-10 > 2:
    #             break
    #         else:
    #             all_fruits[count-10] = [data[i]['x'], data[i]['y']]
    #     else:
    #         obstacles_arr[count] = [data[i]['x'], data[i]['y']]
    #     count += 1

    for i in data:
        if count > 9:
            all_fruits[count-10] = [data[i]['x'], data[i]['y']]
        else:
            obstacles_arr[count] = [data[i]['x'], data[i]['y']]
        count += 1
    
    
    search_fruits = getFruitArr(all_fruits, search_list)
    f.close()
    # print(all_fruits) # Debug
    print("Fruits' Location:{}".format(search_fruits))
    
    waypoints = getPath(tolerance, start_pos, search_fruits, obstacles_arr, search_list)

    # Code For debugging
    print("\nFinal path:")
    print(waypoints)
    print("\n\n")

    return waypoints


################################################### FOR RUNNING FILE INDEPENDENTLY ##############################################
# Main function
def generateWaypointsDebug():
    # Define params:
    start_pos = [0,0]   # start pos
    tolerance = 0.2     # distance when robot will take picture from fruit

    all_fruits = [[0] * 2 for i in range(5)]
    obstacles_arr = [[0] * 2 for i in range(10)]

    # Extracting data from ground truth file
    ground_truth_fname = 'M4_true_map.txt'
    f = open(ground_truth_fname, "r")
    data = json.loads(f.read())
    count = 0

    # for i in data:
    #     if count > 9:
    #         if count-10 > 2:
    #             break
    #         else:
    #             all_fruits[count-10] = [data[i]['x'], data[i]['y']]
    #     else:
    #         obstacles_arr[count] = [data[i]['x'], data[i]['y']]
    #     count += 1

    for i in data:
        if count > 9:
            all_fruits[count-10] = [data[i]['x'], data[i]['y']]
        else:
            obstacles_arr[count] = [data[i]['x'], data[i]['y']]
        count += 1

    search_list = read_search_list()
    search_fruits = getFruitArr(all_fruits, search_list)
    f.close()
    print(all_fruits) # Debug
    print("Fruits' Location:{}".format(search_fruits))   # Debug
    
    waypoints = getPath(tolerance, start_pos, search_fruits, obstacles_arr, search_list)

    # Code For debugging
    print("\nFinal path:")
    print(waypoints)
    print("\n\n")

    return None

generateWaypointsDebug()