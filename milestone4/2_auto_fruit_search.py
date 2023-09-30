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
from network.scripts.detector import Detector
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

# # D*Lite
# from d_star_lite import DStarLite

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
    lv, rv = ppi.set_velocity([0, np.sign(turn_angle_cw - turn_angle_ccw)], turning_tick=wheel_vel_ang, time=turn_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay
    drive_meas = measure.Drive(lv,rv,turn_time)
    robot_pose = get_robot_pose(drive_meas)
    
    # after turning, drive straight to the waypoint
    # print(waypoint)
    # print(robot_pose)
    robot_to_waypoint_distance = np.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1])
    drive_time = robot_to_waypoint_distance / (scale * wheel_vel_lin)
    print("Driving for {:.2f} seconds".format(drive_time))
    lv,rv = ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay
    ####################################################
    drive_meas = measure.Drive(lv,rv,drive_time)
    robot_pose = get_robot_pose(drive_meas)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    # turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    # print("Turning for {:.2f} seconds".format(turn_time))
    # lv,rv = ppi.set_velocity([0, np.sign(turn_angle_cw - turn_angle_ccw)], turning_tick=wheel_vel_ang, time=turn_time)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
    # turning_drive_meas = measure.Drive(lv,rv,turn_time)
    # robot_pose = get_robot_pose(turning_drive_meas)

    # # grp4 special
    # baseline = 11e-2
    # turn_angle = np.pi/6
    # turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    # print("Turning for {:.2f} seconds".format(turn_time))
    # print("Turn angle was :",turn_angle*180/np.pi)
    # lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
    # turning_drive_meas = measure.Drive(lv,rv,turn_time)
    # robot_pose = get_robot_pose(turning_drive_meas)

    # baseline = 10e-2
    # turn_angle = np.pi/3
    # turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    # lv,rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel_ang, time=turn_time)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
    # turning_drive_meas = measure.Drive(lv,rv,turn_time)
    # robot_pose = get_robot_pose(turning_drive_meas)

    # baseline = 11e-2
    # turn_angle = np.pi/6
    # turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    # print("Turning for {:.2f} seconds".format(turn_time))
    # print("Turn angle was :",turn_angle*180/np.pi)
    # lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
    # turning_drive_meas = measure.Drive(lv,rv,turn_time)
    # robot_pose = get_robot_pose(turning_drive_meas)


    # # after turning, drive straight to the waypoint
    # robot_to_waypoint_distance = np.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1])
    # drive_time = robot_to_waypoint_distance / (scale * wheel_vel_lin)
    # print("Driving for {:.2f} seconds".format(drive_time))
    # print("Drive distance was: ",robot_to_waypoint_distance)
    # lv,rv = ppi.set_velocity([1,0], tick=wheel_vel_lin, time=drive_time)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.2) # immediate stop with small delay

    # drive_meas = measure.Drive(lv,rv,drive_time)
    # robot_pose = get_robot_pose(drive_meas)
    
    # ####################################################

    # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    # return robot_pose

# Stay at current position until EKF error values are small
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

        detector_output, _ = d.yolo_detection(img)
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
####################################################
## method 1: open loop position 
    global waypoint
    robot_pose = [0.0,0.0,0.0]

    # obtain angle with respect to x-axis
    robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])
    robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2] # limit from 0 to 360 degree

    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
####################################################
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here  
    # img = np.zeros([240, 320, 3], dtype=np.uint8)
    # img = ppi.get_image()

    # lms, _ = aruco_det.detect_marker_positions(img)
    # ekf.predict(drive_meas)
    # ekf.update(lms)
    # temp_robot_pose = ekf.robot.state
    # if temp_robot_pose[2][0]<0:
    #     temp_robot_pose[2][0] = temp_robot_pose[2][0] + 2*np.pi
    # if temp_robot_pose[2][0]>2*np.pi:
    #     temp_robot_pose[2][0] = temp_robot_pose[2][0] - 2*np.pi
    # robot_pose = [temp_robot_pose[0][0],temp_robot_pose[1][0],temp_robot_pose[2][0]]
    # print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)

    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.209')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    args, _ = parser.parse_known_args()

    ppi = Alphabot(args.ip,args.port)

    file_output = None
    output = dh.OutputWriter('lab_output')
    pred_fname = ''

    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # args.ckpt = "network/scripts/model/best.pt"
    args.ckpt = "network/scripts/model/yolov8_model.pt"
    d = Detector(args.ckpt)

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
    
    print('Fruit list:\n {}\n'.format(fruits_list))
    print('Fruit true pos:\n {}\n'.format(fruits_true_pos))
    print('Aruco true pos:\n {}\n'.format(aruco_true_pos))
    print('Search list:\n {}\n'.format(search_list))

    global robot_pose
    robot_pose = [0.,0.,0.]
    
    # wheel_vel_ang = 20
    # baseline = 11e-2
    # for i in range(10):
    #     turn_angle = np.pi/5
    #     turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    #     lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
    #     ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.2) # immediate stop with small delay
    #     turning_drive_meas = measure.Drive(lv,rv,turn_time)
    #     robot_pose = get_robot_pose(turning_drive_meas)
    # robot_pose_rounded = [round(robot_pose[0]*5)/5, round(robot_pose[1]*5)/5, robot_pose[2]] # Rounding to the nearest 0.2m

    ########################################   A* CODE INTEGRATED ##################################################
    # Get array of waypoints
    waypoints = wp.generateWaypoints()
    for waypoint_progress in range(3):
        # Extract current waypoint
        current_waypoint = waypoints[waypoint_progress]

        # Find current robot pos
        # if waypoint_progress != 0:
        #     robot_pose = get_robot_pose(drive_meas)            
        # print(robot_pose)
        path = pathFind.main(robot_pose, current_waypoint)
        print(path)

        # Drive along path
        for sub_waypoint in path:
            # Drive to segmented waypoints
            global waypoint
            waypoint = sub_waypoint
            print("    ")
            print(sub_waypoint)
            print("    ")
            drive_to_point(sub_waypoint, robot_pose)
            
            # Survey by turning left and right (integrated in drive_to_point)
            # survey()

            # Update robot pos
            # robot_pose = get_robot_pose(drive_meas)

    robot_pose = localize(robot_pose, current_waypoint)













    # ########################################   Drive to Fruit 0 ##################################################
    # d_star = DStarLite([0],[0])
    # _, waypoints_x, waypoints_y = d_star.out(robot_pose[0], robot_pose[1], 0, search_list, fruits_true_pos, aruco_true_pos, fruits_list)

    # waypoints = np.zeros((0,2))
    # for i in range(len(waypoints_x)):
    #     waypoints = np.append(waypoints, [[waypoints_x[i], waypoints_y[i]]], axis = 0)

    # # Removing 1st and last waypoints
    # waypoints = np.delete(waypoints,0,0)
    # waypoints = np.delete(waypoints,-1,0)

    # print(waypoints)
    
    # for i in range(len(waypoints)):
    #     waypoint = [waypoints[i][0], waypoints[i][1]]
    #     robot_pose = drive_to_point(waypoint, robot_pose)
    #     if (i+1)%2 == 0:
    #         robot_pose = localize(robot_pose,waypoint)

        
    #     print("Outside Robot pose :",robot_pose)
    # time.sleep(3) # 3 second delay
    # # robot_pose = localize(robot_pose,waypoint)
    # robot_pose_rounded = [round(robot_pose[0]*5)/5, round(robot_pose[1]*5)/5, robot_pose[2]] # Rounding to the nearest 0.2m
    # ########################################   Drive to Fruit 1 ##################################################
    # d_star = DStarLite([0],[0])
    # _, waypoints_x, waypoints_y = d_star.out(robot_pose_rounded[0], robot_pose_rounded[1], 1, search_list, fruits_true_pos, aruco_true_pos, fruits_list)

    # waypoints = np.zeros((0,2))
    # for i in range(len(waypoints_x)):
    #     waypoints = np.append(waypoints, [[waypoints_x[i], waypoints_y[i]]], axis = 0)

    # # Removing 1st and last waypoints
    # waypoints = np.delete(waypoints,0,0)
    # waypoints = np.delete(waypoints,-1,0)

    # print(waypoints)
    
    # for i in range(len(waypoints)):
    #     waypoint = [waypoints[i][0], waypoints[i][1]]
    #     robot_pose = drive_to_point(waypoint, robot_pose)
    #     if (i+1)%2 == 0:
    #         robot_pose = localize(robot_pose,waypoint)
    #     print("Outside Robot pose :",robot_pose)

    # time.sleep(3) # 3 second delay
    # # robot_pose = localize(robot_pose,waypoint)
    # robot_pose_rounded = [round(robot_pose[0]*5)/5, round(robot_pose[1]*5)/5, robot_pose[2]] # Rounding to the nearest 0.2m

    # ########################################   Drive to Fruit 2 ##################################################
    # d_star = DStarLite([0],[0])
    # _, waypoints_x, waypoints_y = d_star.out(robot_pose_rounded[0], robot_pose_rounded[1], 2, search_list, fruits_true_pos, aruco_true_pos, fruits_list)

    # waypoints = np.zeros((0,2))
    # for i in range(len(waypoints_x)):
    #     waypoints = np.append(waypoints, [[waypoints_x[i], waypoints_y[i]]], axis = 0)

    # # Removing 1st and last waypoints
    # waypoints = np.delete(waypoints,0,0)
    # waypoints = np.delete(waypoints,-1,0)

    # print(waypoints)
    
    # for i in range(len(waypoints)):
    #     waypoint = [waypoints[i][0], waypoints[i][1]]
    #     robot_pose = drive_to_point(waypoint, robot_pose)
    #     # print("Outside Robot pose :",robot_pose)

    