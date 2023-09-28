# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure

# Waypoints GUI
from WaypointNavigationGUI import WaypointNavigationGUI

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
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

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
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel_lin = 30 # tick to move the robot
    wheel_vel_ang = 25
    
    # turn towards the waypoint
    robot_to_waypoint_angle = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]) # Measured from x-axis (theta=0)
    robot_to_waypoint_angle = (robot_to_waypoint_angle + 2*np.pi) if (robot_to_waypoint_angle < 0) else robot_to_waypoint_angle

    turn_angle = robot_to_waypoint_angle - robot_pose[2]
    if (turn_angle == np.pi) or (turn_angle == -np.pi):
        turn_angle = np.pi
        # Dummy values to turn ccw
        turn_angle_ccw = 1
        turn_angle_cw = 0
    elif turn_angle < 0:
        turn_angle_cw = abs(turn_angle)
        turn_angle_ccw = turn_angle + 2*np.pi
        turn_angle = turn_angle_cw if (turn_angle_cw < turn_angle_ccw) else turn_angle_ccw
    elif turn_angle > 0:
        turn_angle_cw = 2*np.pi - turn_angle
        turn_angle_ccw = turn_angle
        turn_angle = turn_angle_cw if (turn_angle_cw < turn_angle_ccw) else turn_angle_ccw
    else: # turn_angle = 0 case
        # Dummy values to not turn
        turn_angle_ccw = 0
        turn_angle_cw = 0

    # Baselines
    if turn_angle <=0.8: # ~45deg
        baseline = 11e-2
    elif turn_angle <=1.6: # ~90deg
        baseline = 9.5e-2
        if np.sign(turn_angle_ccw - turn_angle_cw) < 0:
            baseline = 9e-2 #ccw
    elif turn_angle <=2.4: # ~135deg
        baseline = 8.6e-2
        if np.sign(turn_angle_ccw - turn_angle_cw) < 0:
            baseline = 8.3e-2
    elif turn_angle <=3.2: # ~180deg
        baseline = 8.3e-2

    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, np.sign(turn_angle_cw - turn_angle_ccw)], turning_tick=wheel_vel_ang, time=turn_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay

    # after turning, drive straight to the waypoint
    robot_to_waypoint_distance = np.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1])
    drive_time = robot_to_waypoint_distance / (scale * wheel_vel_lin)
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

def get_robot_pose(waypoint, robot_pose):
    robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]) # Measured from x-axis (theta=0)
    robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2]

    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.119')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos) 

    global robot_pose
    robot_pose = [0.0,0.0,0.0]

    while True:
        waypoints = WaypointNavigationGUI()
        print(waypoints)
        
        for i in range(len(waypoints)):
            waypoint = [waypoints[i][0], waypoints[i][1]]
            drive_to_point(waypoint, robot_pose)
            get_robot_pose(waypoint, robot_pose)

