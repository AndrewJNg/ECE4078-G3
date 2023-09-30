# Import modules
import pygame
import math
# from queue import PriorityQueue
import json
import ast
import numpy as np

def getShortestPath(start_pos, fruits_arr):
    """
    Returns:
        - `1D arr` shortest path index

    Param: 
        - start_pos
        - `2D arr` with 3 fruits position, 
    """
    # Possible paths:
    # 1 -> 2 -> 3
    # 1 -> 3 -> 2
    # 2 -> 1 -> 3
    # 2 -> 3 -> 1
    # 3 -> 1 -> 2
    # 3 -> 2 -> 1
    
    # Format of path variable: [[fruit1,fruit2],[dist], [fruit1,fruit2],[dist],...]
    path = [[0]*2 for i in range(6)]
    # print(path)
    count = 0
    min_dist = 100000
    shortestPath = [-1, -1, -1]
    dist = 0
    for fruit_1 in range(0,3):
        # Get position of start_pos and fruit 1
        x0, y0 = start_pos[0], start_pos[1]
        x1, y1 = fruits_arr[fruit_1][0], fruits_arr[fruit_1][1]
        dist_1 = math.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1))
        for fruit_2 in range(0,3):
            if fruit_2 != fruit_1:
                # Get position of fruit 2
                x2, y2 = fruits_arr[fruit_2][0], fruits_arr[fruit_2][1]

                # Calculate distance for from fruit 1 to fruit 2
                dist_2 = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
                for fruit_3 in range(0,3):
                    if fruit_3 != fruit_2 and fruit_3 != fruit_1:
                        # print('start -> {} -> {} -> {}'.format(fruit_1,fruit_2,fruit_3)) # Prints order of fruit pathing

                        # Get fruit 3 position
                        x3, y3 = fruits_arr[fruit_3][0], fruits_arr[fruit_3][1]

                        # Calculate distance from fruit 2 to fruit 3
                        dist_3 = math.sqrt((x2-x3)*(x2-x3) + (y2-y3)*(y2-y3))

                        path[count][0] = [fruit_1, fruit_2, fruit_3]
                        dist = dist_1 + dist_2 + dist_3
                        path[count][1] = dist
                        count += 1

                        # Track min dist
                        if dist < min_dist:
                            min_dist = dist
                            shortestPath = [fruit_1, fruit_2, fruit_3]
    return shortestPath
    



def getAccuratePath(tolerance, start_pos, fruits_arr, obstacles_arr):
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
        visit_pos_arr['right'] = [round(x0+0.4,1), y0]
        visit_pos_arr['up'] = [x0, round(y0+0.4,1)]
        visit_pos_arr['left'] = [round(x0-0.4,1), y0]
        visit_pos_arr['down'] = [x0, round(y0-0.4,1)]
        visit_pos_arr['upright'] = [round(x0+0.4,1), round(y0+0.4,1)]
        visit_pos_arr['upleft'] = [round(x0-0.4,1), round(y0+0.4,1)]
        visit_pos_arr['downleft'] = [round(x0-0.4,1), round(y0-0.4,1)]
        visit_pos_arr['downright'] = [round(x0+0.4,1), round(y0-0.4,1)]
        print('\nFruit {}: {}'.format(index, visit_pos_arr))  # Debug
        # Create a list to store positions to remove
        positions_to_remove = []
        # Check if obstacle is in each location
        for obstacle_idx, obstacle_pos in enumerate(obstacles_arr):
            x_o, y_o = obstacle_pos[0], obstacle_pos[1]
            for dir, pos in visit_pos_arr.items():
                if x_o == pos[0] and y_o == pos[1]:
                    print('Obstacle {} and visit_pos clashed at {}'.format(obstacle_idx, pos))    # Debug
                    positions_to_remove.append(dir)
        
        # Remove the positions that clashed with obstacles
        for dir in positions_to_remove:
            visit_pos_arr.pop(dir)

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

        min_dir_arr.append(min_dir) # TODO comment
        
    # print('\nWhere robot is relative to fruit: {}\n'.format(min_dir_arr))   # Debug
    return final_visit_pos

# TODO Implement
# def read_search_list():
#     """Read the search order of the target fruits

#     @return: search order of the target fruits
#     """
#     search_list = []
#     with open('search_list.txt', 'r') as fd:
#         fruits = fd.readlines()

#         for fruit in fruits:
#             search_list.append(fruit.strip())

#     return search_list
      
# Based on search order, generate fruit_arr containing fruit position
def getFruitArr(all_fruits, search_list_file_name):
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
    return search_fruits

# Main function
def generateWaypoints():
    # Define params:
    start_pos = [0,0]   # start pos
    tolerance = 0.4     # distance when robot will take picture from fruit

    all_fruits = [[0] * 2 for i in range(3)]
    obstacles_arr = [[0] * 2 for i in range(10)]\

    # Extracting data from ground truth file
    ground_truth_fname = 'M4_true_map.txt'
    f = open(ground_truth_fname, "r")
    data = json.loads(f.read())
    count = 0
    for i in data:
        if count > 9:
            if count-10 > 2:
                break
            else:
                all_fruits[count-10] = [data[i]['x'], data[i]['y']]
        else:
            obstacles_arr[count] = [data[i]['x'], data[i]['y']]
        count += 1
    
    search_fruits = getFruitArr(all_fruits, 'search_list.txt')
    f.close()
    print("Fruits' Location:{}".format(search_fruits))   # Debug


    # fruit_order = getShortestPath(start_pos, search_fruits)
    waypoints = getAccuratePath(tolerance, start_pos, search_fruits, obstacles_arr)

    # Code For debugging
    # shortest_path_str = [getFruitName(shortestPath[0]), getFruitName(shortestPath[1]), getFruitName(shortestPath[2])]
    # print("Shortest path:\n{}".format(shortest_path_str))  # Debug
    # print("\nFinal path:")
    # print(waypoints)

    ## To Output into file (Not Necessary Anymore)
    # file_output = {}
    # for index, path in enumerate(finalPath):
    #     file_output['checkpoint_' + str(index+1)] = {"x": finalPath[index][0], "y":finalPath[index][1], "theta":finalPath[index][2]}

    # with open('waypoint.txt', 'w') as f:
    #     json.dump(file_output, f, indent=4)

    return waypoints
    

