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

from network.scripts.detector import Detector
import util.DatasetHandler as dh # save/load functions


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


###############################################################################################################################
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
    
    ########################################################################################################
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

    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, np.sign(turn_angle_cw - turn_angle_ccw)], turning_tick=wheel_vel_ang, time=turn_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay

    # after turning, drive straight to the waypoint
    print(waypoint)
    print(robot_pose)
    robot_to_waypoint_distance = np.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1])
    drive_time = robot_to_waypoint_distance / (scale * wheel_vel_lin)
    print("Driving for {:.2f} seconds".format(drive_time))

    lv,rv = ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay
    ####################################################
    drive_meas = measure.Drive(lv,rv,drive_time)
    robot_pose = get_robot_pose(drive_meas)
    robot_pose = localize(robot_pose, waypoint)
    ####################################################

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    '''
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    kv_p = 1 
    kw_p = 5 
    
    threshold = 0.2

    distance_to_goal = get_distance_robot_to_goal(robot_pose, waypoint)
    desired_heading  = get_angle_robot_to_goal(robot_pose, waypoint)
    

    while( desired_heading > threshold ):
        kv_p = 0                
        v_k = kv_p * distance_to_goal
        w_k = kw_p * desired_heading
        
        
        # Apply control to robot
        ppi.set_velocity([v_k, w_k])
        robot_pose = get_robot_pose()
        desired_heading = get_angle_robot_to_goal(robot_pose, waypoint) 

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
        distance_to_goal = get_distance_robot_to_goal(robot_pose, waypoint)
        desired_heading  = get_angle_robot_to_goal(robot_pose, waypoint)
    '''
    

# def get_robot_pose(waypoint, robot_pose):
# ####################################################
# ## method 1: open loop position 
#     robot_pose = [0.0,0.0,0.0]

#     # obtain angle with respect to x-axis
#     robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])
#     robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2] # limit from 0 to 360 degree

#     robot_pose[0] = waypoint[0]
#     robot_pose[1] = waypoint[1]
# ####################################################
# ## method 2: using EKF

#     return robot_pose

def localize(robot_pose, waypoint):
    rms_error = 10
    baseline = 11e-2
    turn_angle = 0.45
    wheel_vel_ang = 16
    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))

    lms = [0]
    continue_robot = 1
    while continue_robot:
        lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
        ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
        turning_drive_meas = measure.Drive(lv,rv,turn_time)
        # robot_pose = get_robot_pose(turning_drive_meas)
        time.sleep(0.2)
        img = np.zeros([240, 320, 3], dtype=np.uint8)
        img = ppi.get_image()
        lms, _ = aruco_det.detect_marker_positions(img)
        ekf.predict(turning_drive_meas)
        ekf.update(lms)

        detector_output, _ = d.detect_single_image(img)
        if len(detector_output)>0:
            print("seen fruit")
            file_output = (detector_output, ekf)
            pred_fname = output.write_image(file_output[0],file_output[1])
            print("saved as: ",pred_fname)

        temp_robot_pose = ekf.robot.state
        if temp_robot_pose[2][0]<0:
            temp_robot_pose[2][0] = temp_robot_pose[2][0] + 2*np.pi
        if temp_robot_pose[2][0]>2*np.pi:
            temp_robot_pose[2][0] = temp_robot_pose[2][0] - 2*np.pi
        robot_pose = [temp_robot_pose[0][0],temp_robot_pose[1][0],temp_robot_pose[2][0]]

        rms_error = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
        print("\nrms_error: ",rms_error)
        print("lms count: ",len(lms))
        if (rms_error<=0.2):
            continue_robot = 0
    return robot_pose


def get_robot_pose(drive_meas):
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here  
    img = np.zeros([240, 320, 3], dtype=np.uint8)
    img = ppi.get_image()

    lms, _ = aruco_det.detect_marker_positions(img)
    ekf.predict(drive_meas)
    ekf.update(lms)
    temp_robot_pose = ekf.robot.state
    if temp_robot_pose[2][0]<0:
        temp_robot_pose[2][0] = temp_robot_pose[2][0] + 2*np.pi
    if temp_robot_pose[2][0]>2*np.pi:
        temp_robot_pose[2][0] = temp_robot_pose[2][0] - 2*np.pi
    robot_pose = [temp_robot_pose[0][0],temp_robot_pose[1][0],temp_robot_pose[2][0]]
    print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)
    return robot_pose

###############################################################################################################################
"""
	Compute Euclidean distance between the robot and the goal location
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
"""
def get_distance_robot_to_goal(robot_state=np.zeros(3), goal=np.zeros(3)):
    goal = np.array([goal,0])
    goal.reshape(1,3)

    if goal.shape[0] < 3:
        goal = np.hstack((goal, np.array([0])))

    x_goal, y_goal,_ = goal
    x, y,_ = robot_state
    rho = np.hypot(x_goal - x, y_goal - y) 
    return rho


############################################################################################################################################################
# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    # update robot characteristics
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    args.ckpt = "network/scripts/model/yolov8_model.pt"
    d = Detector(args.ckpt)
    
    file_output = None
    output = dh.OutputWriter('lab_output')
    pred_fname = ''

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    aruco_det = aruco.aruco_detector(robot)
    ekf = EKF(robot)

    lms = []
    for i,lm in enumerate(aruco_true_pos):
        measurement_lm = measure.Marker(np.array([[lm[0]],[lm[1]]]),i+1)
        lms.append(measurement_lm)
    ekf.add_landmarks(lms)

    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    print(fruits_list)
    print(fruits_true_pos)
    print(aruco_true_pos)
    print(search_list)

    global waypoint
    waypoint = [0.0,0.0]

    global robot_pose
    robot_pose = [0.0,0.0,0.0]

    while True:

        # provide waypoints in a list format, example shown below
        # waypoints = [[0.4,0],[0.8,0],[0,0]]
        waypoints = [[0.4,0],[0.8,0],[0,0]]

        # travel through all waypoints one after the other
        for sub_waypoint in waypoints:
            # robot drives to the waypoint
            drive_to_point(sub_waypoint,robot_pose)

            print("Finished driving to waypoint: {}; New robot pose: {}".format(sub_waypoint,robot_pose))
            ppi.set_velocity([0, 0])



