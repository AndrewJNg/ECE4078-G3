# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time
import math
import TargetPoseEst

# import neural network detector
from network.scripts.detector import Detector # modified for yolov8

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import util.DatasetHandler as dh # save/load functions

# import utility functions
sys.path.insert(0, "util")
from pibot import Alphabot
import measure as measure
import generateWaypoints as wp
import pathFind


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


########################################################################################################
# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint):
    global robot_pose
    
    ####################################################
    # rotate robot to turn towards the waypoint
    robot_to_waypoint_angle = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]) # Measured from x-axis (theta=0)
    turn_angle = robot_to_waypoint_angle - robot_pose[2]
    robot_turn(turn_angle)

    ####################################################
    # after turning, drive straight to the waypoint
    robot_straight(robot_to_waypoint_distance = np.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1]))
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))



########################################################################################################

def robot_turn(turn_angle=0,wheel_vel_lin=30,wheel_vel_ang = 25):
    """
    purpose:
    ### enable the robot to turn a certain angle automatically
   

    input: 
    ### turn_angle: angle to rotate on the spot radians (negative value means turn right)

    output:
    ### physical robot: robot would rotate by turn_angle amount
    """
    global robot_pose
    wheel_vel_lin = 30 # tick to move the robot
    wheel_vel_ang = 25
    
    # limit angle between -180 to 180 degree (suitable for robot turning)
    turn_angle = (turn_angle) % (2*np.pi) 
    turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

    turn_time = abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))
    print("Turning for {:.2f} seconds".format(turn_time))

    lv, rv = ppi.set_velocity([0, np.sign(turn_angle)], turning_tick=wheel_vel_ang, time=turn_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    drive_meas = measure.Drive(lv,rv,turn_time)
    robot_pose = get_robot_pose(drive_meas)

########################################################################################################

def robot_straight(robot_to_waypoint_distance=0,wheel_vel_lin=30,wheel_vel_ang = 25):
    global robot_pose
    wheel_vel_lin = 30 # tick to move the robot
    wheel_vel_ang = 25

    drive_time = robot_to_waypoint_distance / (scale * wheel_vel_lin)
    print("Driving for {:.2f} seconds".format(drive_time))

    lv,rv = ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    drive_meas = measure.Drive(lv,rv,drive_time)
    robot_pose = get_robot_pose(drive_meas)

########################################################################################################

def get_robot_pose(drive_meas):
####################################################
    ## method 1: open loop position 
    '''
    global waypoint
    global robot_pose
    # robot_pose = [0.0,0.0,0.0]

    # obtain angle with respect to x-axis
    robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])
    robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2] # limit from 0 to 360 degree

    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
    '''
####################################################
    ## method 2: Using SLAM through EKF
    # '''
    img = ppi.get_image()

    landmarks, _ = aruco_det.detect_marker_positions(img)
    ekf.predict(drive_meas)
    ekf.update(landmarks) 
    robot_pose = ekf.robot.state

    print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)

    
    # cv2.imshow('Predict',  img)
    # cv2.waitKey(0)
    # '''

    return robot_pose

####################################################################################################

def localize(waypoint):
    global robot_pose
    rms_error = 10
    turn_angle = 0.45
    wheel_vel_ang = 16
    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))

    landmarks = [0]
    continue_robot = 1
    while continue_robot:
        lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
        ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.5) # stop with delay
        turning_drive_meas = measure.Drive(lv,rv,turn_time)
        robot_pose = get_robot_pose(turning_drive_meas)

        img = ppi.get_image()
        landmarks, aruco_img = aruco_det.detect_marker_positions(img)


        ## TODO print out landmarks array, and check for how many markers it sees, try to get accurate result, and then leave loop
        ekf.predict(turning_drive_meas)
        ekf.update(landmarks)
        robot_pose = ekf.robot.state
        print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)

        cv2.imshow('Predict',  aruco_img)
        cv2.waitKey(0)
        

        detector_output, img_yolov = yolov.detect_single_image(img)

        cv2.imshow('Predict',  img_yolov)
        cv2.waitKey(0)
        if len(detector_output)>0:
            print("seen fruit")
            # file_output = (detector_output, ekf)
            # pred_fname = output.write_image(file_output[0],file_output[1])
            # print("saved as: ",pred_fname)

        robot_pose = ekf.robot.state
        rms_error = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
        print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)
        print("\nrms_error: ",rms_error)
        print("landmarks count: ",len(landmarks))

        if (rms_error<=0.2):
            continue_robot = 0
    return robot_pose

####################################################################################################
# main loop
if __name__ == "__main__":
    # arguments for starting command
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.209')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    # robot startup with given variables
    ppi = Alphabot(args.ip,args.port)
    
    # file output location for predicted output by neural network
    file_output = None
    output = dh.OutputWriter('lab_output')
    pred_fname = ''


    # obtain robot settings for camera and motor (found by calibration)
    global dist_coeffs
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    
    global camera_matrix
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    
    global scale
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    
    global baseline
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # neural network file location
    args.ckpt = "network/scripts/model/yolov8_model_best.pt"
    yolov = Detector(args.ckpt)

####################################################
# set up all EKF using given values
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    aruco_det = aruco.aruco_detector(robot, marker_length = 0.06)
    ekf = EKF(robot)

    landmarks = []
    for i,landmark in enumerate(aruco_true_pos):
        measurement_landmark = measure.Marker(np.array([[landmark[0]],[landmark[1]]]),i+1)
        landmarks.append(measurement_landmark)
    ekf.add_landmarks(landmarks)

####################################################
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    
    print('Fruit list:\n {}\n'.format(fruits_list))
    print('Fruit true pos:\n {}\n'.format(fruits_true_pos))
    print('Aruco true pos:\n {}\n'.format(aruco_true_pos))
    print('Search list:\n {}\n'.format(search_list))

    global robot_pose
    robot_pose = [0.,0.,0.]
    
    # movement test
    '''
    robot_turn(-3,wheel_vel_lin=30,wheel_vel_ang = 25)
    robot_straight(0.4,wheel_vel_lin=30,wheel_vel_ang = 25)
    '''

########################################   A* CODE INTEGRATED ##################################################
    '''
    waypoints = wp.generateWaypoints()
    print(waypoints)
    for waypoint_progress in range(3):
        # Extract current waypoint
        current_waypoint = waypoints[waypoint_progress]        
        path = pathFind.main(robot_pose, current_waypoint,[])
        print(path)

        for sub_waypoint in path:
            # Drive to segmented waypoints
            global waypoint
            waypoint = sub_waypoint
            print("    ")
            print("target: "+str(sub_waypoint))
            drive_to_point(sub_waypoint)

            localize(waypoint)
    '''

    # bare minimum waypoint
    # '''
    waypoints = [[0.2,0],[0.4,0],[0.4,0.2],[0.6,0.2],[0.8,0.2],[1.0,0.2],[1.2,0.2],[1.2,0] ]
    for sub_waypoint in waypoints :
        # Drive to segmented waypoints
        global waypoint
        print("    ")
        print("target: "+str(sub_waypoint))
        waypoint = sub_waypoint
        drive_to_point(sub_waypoint)
        localize(waypoint)
    # '''




