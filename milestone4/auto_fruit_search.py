# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure


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
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
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
    kv_p = 1 
    kw_p = 5 
    
    threshold = 0.2

    distance_to_goal = get_distance_robot_to_goal(robot_pose, [waypoint,0])
    desired_heading  = get_angle_robot_to_goal(robot_pose, [waypoint,0])
    

    while( desired_heading > threshold ):
        kv_p = 0                
        v_k = kv_p * distance_to_goal
        w_k = kw_p * desired_heading
        
        
        # Apply control to robot
        ppi.set_velocity([v_k, w_k])
        robot_pose = get_robot_pose()
        desired_heading = get_angle_robot_to_goal(robot_pose, [waypoint,0]) 

    kv_p = 1 
    kw_p = 5 

    while (distance_to_goal>threshold): 
        
        #TODO 3: Compute the new control input ---------------------------------
        v_k = kv_p * distance_to_goal
        w_k = kw_p * desired_heading
                    
        #TODO 4: Update errors ---------------------------------------------------
        # updated errors
        ppi.set_velocity([v_k, w_k])
        robot_pose = get_robot_pose()
        distance_to_goal = get_distance_robot_to_goal(robot_pose, [waypoint,0])
        desired_heading  = get_angle_robot_to_goal(robot_pose, [waypoint,0])
        

    # wheel_vel = 10 # tick to move the robot
    # angular_vel = 5

    # turn towards the waypoint
    # desired_heading  = get_angle_robot_to_goal(robot_pose, [waypoint,0])
    # turn_time = 0.0 # replace with your calculation

    # turn_time = 





    # print("Turning for {:.2f} seconds".format(turn_time))
    # ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    
    # # after turning, drive straight to the waypoint
    # drive_time = 0.0 # replace with your calculation
    # print("Driving for {:.2f} seconds".format(drive_time))
    # ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # update the robot pose [x,y,theta]
    robot_pose = [0.0,0.0,0.0] # replace with your calculation
    # self.init_ekf(args.calib_dir, args.ip)
    # EKF.predict



    ####################################################

    return robot_pose

def get_distance_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
	"""
	Compute Euclidean distance between the robot and the goal location
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""

	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y,_ = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y

	rho = np.hypot(x_diff, y_diff)

	return rho


def get_angle_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
	"""
	Compute angle to the goal relative to the heading of the robot.
	Angle is restricted to the [-pi, pi] interval
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""

	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y, theta = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y
    
	alpha = clamp_angle(np.arctan2(y_diff, x_diff) - theta)

	return alpha


def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
	"""
	Restrict angle to the range [min, max]
	:param rad_angle: angle in radians
	:param min_value: min angle value
	:param max_value: max angle value
	"""

	if min_value > 0:
		min_value *= -1

	angle = (rad_angle + max_value) % (2 * np.pi) + min_value

	return angle

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break