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
import shutil # python package for file operations

# import utility functions
sys.path.insert(0, "util")
from util.pibot import Alphabot
import util.measure as measure
import generateWaypoints as wp
import pathFind
import pygame

################################################################### USER INTERFACE ###################################################################
class Operate:
    def __init__(self):
        self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        # self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = True
        # self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        # self.control_clock = time.time()
        # initialise images
        # self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    """def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas"""
    # camera control
    """def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)"""

    # SLAM with ARUCO markers       
    """def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)"""

    # save images taken by the camera
    """def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'"""

    # wheel and camera calibration for SLAM
    """def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)"""

    # save SLAM map
    """def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False"""

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad)) # M2
        self.put_caption(canvas, caption='Detector (M3)',
                         position=(h_pad, 240+2*v_pad)) # M3
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

def initiate_UI():
    pygame.font.init() 
    global TITLE_FONT
    global TEXT_FONT
    global canvas 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2022 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    global start
    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    return None




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

def image_to_camera_coordinates(bounding_box, camera_matrix, rotation_matrix, translation_vector):
    # Define the 2D bounding box points
    x_min, y_max,width, height = bounding_box
    x_max = x_min - width
    y_min = y_max - height
    # x_min, y_min, x_max, y_max = bounding_box

    # Calculate the center of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Create a homogeneous 2D point
    point_2d = np.array([x_center, y_center, 1.0])

    # Invert the camera matrix to get the camera's extrinsic matrix
    inverse_camera_matrix = np.linalg.inv(camera_matrix)

    # Calculate the 3D point in camera coordinates
    point_3d_camera = np.dot(inverse_camera_matrix, point_2d)

    # Apply the rotation and translation to convert to world coordinates
    point_3d_world = np.dot(rotation_matrix, point_3d_camera) + translation_vector

    return point_3d_world
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
    # print(robot_to_waypoint_angle)
    # print(robot_pose[2])
    # print(turn_angle)
    robot_turn(turn_angle)

    ####################################################
    # after turning, drive straight to the waypoint
    robot_straight(robot_to_waypoint_distance = math.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1]))
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))



########################################################################################################

def robot_turn(turn_angle=0,wheel_vel_lin=30,wheel_vel_ang = 20):
    """
    purpose:
    ### enable the robot to turn a certain angle automatically
   

    input: 
    ### turn_angle: angle to rotate on the spot radians (negative value means turn right)

    output:
    ### physical robot: robot would rotate by turn_angle amount
    """
    global robot_pose
    global baseline
    # wheel_vel_lin = 30 # tick to move the robot
    # wheel_vel_ang = 10
    if abs(turn_angle) <=1.6: # ~90deg
        baseline = 10.6e-2

    elif abs(turn_angle) <=3.2: # ~180deg
        baseline = 8.9e-2

    
    # limit angle between -180 to 180 degree (suitable for robot turning)
    turn_angle = (turn_angle) % (2*np.pi) 
    turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

    # make robot turn a certain angle
    turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
    # print("Turning for {:.2f} seconds".format(turn_time))

    lv, rv = ppi.set_velocity([0, np.sign(turn_angle)], turning_tick=wheel_vel_ang, time=turn_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    drive_meas = measure.Drive(lv,rv,turn_time)
    get_robot_pose(drive_meas)

    # visualise
    operate.draw(canvas)
    pygame.display.update()
    
    # return robot_pose
    return None

########################################################################################################

def robot_straight(robot_to_waypoint_distance=0,wheel_vel_lin=30,wheel_vel_ang = 25):
    global robot_pose
    # wheel_vel_lin = 30 # tick to move the robot
    # wheel_vel_ang = 25

    drive_time = (robot_to_waypoint_distance / (scale * wheel_vel_lin) )
    # print("Driving for {:.2f} seconds".format(drive_time))

    lv,rv = ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    drive_meas = measure.Drive(lv,rv,drive_time)
    get_robot_pose(drive_meas)

    # visualise
    operate.draw(canvas)
    pygame.display.update()

    # return robot_pose
    return None

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
    global robot_pose
    global landmarks # NEW Added
    landmarks, detector_output = take_and_analyse_picture()
    ekf.predict(drive_meas)
    # ekf.add_landmarks(landmarks) #TODO not sure if this is needed or not
    ekf.update(landmarks) 

    
    robot_pose = ekf.robot.state.reshape(-1)
    print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")


    return robot_pose, landmarks

####################################################################################################
def take_and_analyse_picture():
    img = ppi.get_image()
    global aruco_img
    landmarks, aruco_img, boundingbox = aruco_det.detect_marker_positions(img)
    # cv2.imshow('Predict',  aruco_img)
    # cv2.waitKey(0)
    # landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    detector_output, img_yolov = yolov.detect_single_image(img)
    print(detector_output)


    # cv2.imshow('Predict',  img_yolov)
    # cv2.waitKey(0)
    # print(aruco_corners[0][0])
    # print(np.mean(aruco_corners[0][0], axis=0)[0])
    '''
    while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) > 20):
        wheel_vel_lin = 30 # tick to move the robot
        wheel_vel_ang = 25
        img = ppi.get_image()

        landmarks, aruco_img = aruco_det.detect_marker_positions(img)
        # global robot_pose

        
        # turn_angle = -0.2
        # # limit angle between -180 to 180 degree (suitable for robot turning)
        # turn_angle = (turn_angle) % (2*np.pi) 
        # turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

        # # make robot turn a certain angle
        # turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
        # print("Turning for {:.2f} seconds".format(turn_time))

        lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel_ang)
        # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) < -20):
        img = ppi.get_image()

        landmarks, aruco_img = aruco_det.detect_marker_positions(img)
        # global robot_pose
        
        # wheel_vel_lin = 30 # tick to move the robot
        # wheel_vel_ang = 25
        
        # turn_angle = 0.2
        # # limit angle between -180 to 180 degree (suitable for robot turning)
        # turn_angle = (turn_angle) % (2*np.pi) 
        # turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

        # # make robot turn a certain angle 
        # turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
        # print("Turning for {:.2f} seconds".format(turn_time))

        lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    '''
    
    # cv2.imshow('Predict',  aruco_img)
    # cv2.waitKey(0)
    # cv2.imshow('Predict',  img_yolov)
    # cv2.waitKey(0)

    detector_output =0
    return landmarks, detector_output
    # return landmarks, detector_output,aruco_corners

def turn_to_fruit(wheel_vel_lin=40,wheel_vel_ang=10):
    img = ppi.get_image()
    detector_output, img_yolov = yolov.detect_single_image(img)
    timer = time.time()
    Continue =1
    while (Continue == 1):
        try:
            if(len(detector_output)>=1):
                print(detector_output[0][1])
                mean = (detector_output[0][1][0]+detector_output[0][1][2])/2


                while((mean-(640/2)) > 20):
                    lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel_ang, time=0.3)
                    ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.3)
                    img = ppi.get_image()
                    detector_output, img_yolov = yolov.detect_single_image(img)
                    mean = (detector_output[0][1][0]+detector_output[0][1][2])/2
                    print(mean)
                
                while((mean-(640/2)) < -20):
                    lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=0.3)
                    ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.3)
                    img = ppi.get_image()
                    detector_output, img_yolov = yolov.detect_single_image(img)
                    mean = (detector_output[0][1][0]+detector_output[0][1][2])/2
                    print(mean)

                ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
                # cv2.imshow('Predict',  img_yolov)
                # cv2.waitKey(0)
                break
            else:
                print("no fruits found")
                break
        except:
            if((time.time()-timer)>10):
                break


    # while((mean-(640/2)) < -20):
    #     img = ppi.get_image()
    #     detector_output, img_yolov = yolov.detect_single_image(img)
    #     lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay


    # while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) > 20):
    #     wheel_vel_lin = 30 # tick to move the robot
    #     wheel_vel_ang = 25
    #     img = ppi.get_image()

    #     landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    #     # global robot_pose

        
    #     # turn_angle = -0.2
    #     # # limit angle between -180 to 180 degree (suitable for robot turning)
    #     # turn_angle = (turn_angle) % (2*np.pi) 
    #     # turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

    #     # # make robot turn a certain angle
    #     # turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
    #     # print("Turning for {:.2f} seconds".format(turn_time))

    #     lv, rv = ppi.set_velocity([0, -1], turning_tick=wheel_vel_ang)
    #     # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    # while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) < -20):
    #     img = ppi.get_image()

    #     landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    #     lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    # while((np.mean(input_bbox, axis=0)[0]-(640/2)) < -20):
    #     img = ppi.get_image()

    #     landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    #     # global robot_pose
        
    #     # wheel_vel_lin = 30 # tick to move the robot
    #     # wheel_vel_ang = 25
        
    #     # turn_angle = 0.2
    #     # # limit angle between -180 to 180 degree (suitable for robot turning)
    #     # turn_angle = (turn_angle) % (2*np.pi) 
    #     # turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

    #     # # make robot turn a certain angle 
    #     # turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
    #     # print("Turning for {:.2f} seconds".format(turn_time))

    #     lv, rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

####################################################################################################
# '''
def localize(waypoint): # turn and call get_robot_pose
    global robot_pose
    # baseline = 11.6e-2
    turn_angle = 2*np.pi/10
    wheel_vel_ang = 10
    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    robot_pose_previous = robot_pose
    print("\nLocalising Now")
    turn_count = 0
    latest_pose = np.zeros((0,3))
    aruco_3_skip_flag = 0
    while turn_count<10:
        lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
        turning_drive_meas = measure.Drive(lv,rv,turn_time)
        ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.8) # immediate stop with small delay
        robot_pose, lms = get_robot_pose(turning_drive_meas)
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        print(turning_drive_meas)
        
        # if lms>2:
        #     aruco_3_skip_flag = 1
        #     break
        ##########################################
        # Save robot poses and aruco markers seen in an array, then choose latest 2 aruco seen position, if available
        latest_pose = np.append(latest_pose,[[robot_pose[0],robot_pose[1],lms]],0)
        ##########################################
        turn_count += 1
        print(turn_count)
        robot_pose_previous = robot_pose
        # rms_error = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
        # print("rms_error: ",rms_error)
        # if (rms_error<0.2):
        #     continue_turning = 0
        
        robot_pose[2] = (robot_pose[2]) % (2*np.pi) 
        robot_pose[2] = robot_pose[2]-2*np.pi if robot_pose[2]>np.pi else robot_pose[2]
        print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")
        
    print(f"Pose after localised : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")
    print("Finished localising")

    ######
    # if aruco_3_skip_flag == 0:
    #     for i in range(len(latest_pose)):
    #         if latest_pose[i,2]>=2:
    #             robot_pose[0] = latest_pose[i,0]
    #             robot_pose[1] = latest_pose[i,1]
    # aruco_3_skip_flag = 0
    ######
    return robot_pose
    # '''
####################################################################################################
'''
def localize(waypoint):
    rms_error = 10
    timer_old = time.time()
    landmarks = [0]
    continue_robot = 1
    global robot_pose
    rms_error = 10
    turn_angle = np.pi/4
    wheel_vel_ang = 16

    landmarks = [0]
    continue_robot = 1
    # landmarks, detector_output = take_and_analyse_picture()
    # print(landmarks[0][0][0])
    # if()
    # if len(landmarks)==1:
        # print(landmarks)
    while continue_robot:
        robot_turn(turn_angle=turn_angle, wheel_vel_ang =wheel_vel_ang)


        # if len(landmarks)==1:
            

        # get_robot_pose(drive_meas)
        # landmarks, detector_output = take_and_analyse_picture()

        # if len(detector_output)>0:
        #     print("seen fruit")
        #     file_output = (detector_output, ekf)
        #     pred_fname = output.write_image(file_output[0],file_output[1])
        #     print("saved as: ",pred_fname)

        rms_error = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
        print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")
        # print("\nrms_error: ",rms_error)
        # print("landmarks count: ",len(landmarks))
        # print(time.time() - timer_old)

        if (rms_error<=0.2):
        # if (rms_error<=0.2 or ((time.time() - timer_old) >20)):
            continue_robot = 0
    # return robot_pose
'''
####################################################################################################
'''
def localize(waypoint):
    global robot_pose
    rms_error = 10
    turn_angle = 0.45
    wheel_vel_ang = 16
    turn_time = turn_angle * ((baseline/2)/(scale*wheel_vel_ang))
    timer_old = time.time()
    landmarks = [0]
    continue_robot = 1
    while continue_robot:
        lv,rv = ppi.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
        ppi.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.5) # stop with delay
        turning_drive_meas = measure.Drive(lv,rv,turn_time)
        get_robot_pose(turning_drive_meas)

        img = ppi.get_image()
        landmarks, aruco_img = aruco_det.detect_marker_positions(img)


        ## TODO print out landmarks array, and check for how many markers it sees, try to get accurate result, and then leave loop
        ekf.predict(turning_drive_meas)
        ekf.update(landmarks)
        temp_pose = ekf.robot.state
        robot_pose = [temp_pose[0].tolist()[0],temp_pose[1].tolist()[0],temp_pose[2].tolist()[0]]
        print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)

        # cv2.imshow('Predict',  aruco_img)
        # cv2.waitKey(0)
        

        detector_output, img_yolov = yolov.detect_single_image(img)

        # cv2.imshow('Predict',  img_yolov)
        # cv2.waitKey(0)
        if len(detector_output)>0:
            print("seen fruit")
            # file_output = (detector_output, ekf)
            # pred_fname = output.write_image(file_output[0],file_output[1])
            # print("saved as: ",pred_fname)

        temp_pose = ekf.robot.state
        robot_pose = [temp_pose[0].tolist()[0],temp_pose[1].tolist()[0],temp_pose[2].tolist()[0]]
        rms_error = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
        print("Get Robot pose :",robot_pose,robot_pose[2]*180/np.pi)
        print("\nrms_error: ",rms_error)
        print("landmarks count: ",len(landmarks))
        print(time.time() - timer_old)

        if (rms_error<=0.5) or ((time.time() - timer_old) >20):
            continue_robot = 0
    # return robot_pose
'''
'''
def estimate_position():
    img = ppi.get_image()
    global aruco_img
    landmarks, aruco_img,bounding_box = aruco_det.detect_marker_positions(img)
    robot_straight()
    print(bounding_box)
    
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1]])  # Replace with your camera rotation matrix

    translation_vector = np.array([0, 0, 0])  # Replace with your camera translation vector
    point_3d = image_to_camera_coordinates(bounding_box[0], camera_matrix, rotation_matrix, translation_vector)
    print("3D Point in Camera Coordinates:", point_3d)

    # ekf.add_landmarks(landmarks) #TODO not sure if this is needed or not
    ekf.update(landmarks) 

    robot_pose = ekf.robot.state.reshape(-1)
    # print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")
    
    cv2.imshow('Predict',  aruco_img)
    cv2.waitKey(0)
'''
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
    print(aruco_true_pos)
    # """
    for i,landmark in enumerate(aruco_true_pos):
        measurement_landmark = measure.Marker(np.array([[landmark[0]],[landmark[1]]]),i+1)
        landmarks.append(measurement_landmark)
    print(landmarks)
    ekf.add_landmarks(landmarks)

####################################################
# Initiate UI
    start = initiate_UI()
    operate = Operate()

####################################################
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    
    # print('Fruit list:\n {}\n'.format(fruits_list))
    # print('Fruit true pos:\n {}\n'.format(fruits_true_pos))
    # print('Aruco true pos:\n {}\n'.format(aruco_true_pos))
    # print('Search list:\n {}\n'.format(search_list))

    global robot_pose
    robot_pose = [0.,0.,0.]
    ppi.set_velocity([0, 0]) # stop with delay
    
    # global aruco_img
    # movement test
    # '''
    # robot_turn(3,wheel_vel_lin=30,wheel_vel_ang = 25)
    # robot_straight(0.4,wheel_vel_lin=30,wheel_vel_ang = 25)
    # robot_turn(3,wheel_vel_lin=30,wheel_vel_ang = 25)
    # robot_straight(0.4,wheel_vel_lin=30,wheel_vel_ang = 25)
    
    # robot_turn(3,wheel_vel_lin=30,wheel_vel_ang = 25)
    # robot_straight(0.4,wheel_vel_lin=30,wheel_vel_ang = 25)
    # take_and_analyse_picture()
    # waypoint = [0,0]
    # localize(waypoint)
    # '''
    # estimate_position()
    # take_and_analyse_picture()
    # turn_to_fruit()

########################################   A* CODE INTEGRATED ##################################################
    '''
    while True:
        try:
            print()
            print("next")
            x = input("input, x: ")
            y = input("input, y: ")
            sub_waypoint = [float(x),float(y)]
            drive_to_point(sub_waypoint)
        except:
            print("enter again")
    '''
    # '''
    waypoints = wp.generateWaypoints(search_list)
    
    for waypoint_progress in range(3):
        # global waypoint
        # Extract current waypoint
        # print(waypoints)
        current_waypoint = waypoints[waypoint_progress]
        # print("Target waypoint:")  
        # print(current_waypoint)      
        if waypoint_progress == 0:
            current_start_pos = [0,0]
        else: 
            current_start_pos = waypoints[waypoint_progress-1]
        path = pathFind.main(current_start_pos, current_waypoint,[])
        path.pop(0)
        path.pop(0)
        path.append(path[-1]) # NEW Added last sub-waypoint again
        print(path)

        # print("localising")
        # localize(path[0])
        # print("localise done")
        for i, sub_waypoint in enumerate(path, 3):
            # Drive to segmented waypoints
            # waypoint = sub_waypoint
            print()
            print()
            print("    ")
            print("target: "+str(sub_waypoint))
            # print("before_POSE",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)
            drive_to_point(sub_waypoint)
            print("after_POSE",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)
        
            # cv2.imshow('Predict',  aruco_img)
            # cv2.waitKey(0)
            if (i+1)%20 == 0:
                print("localising")
                localize(sub_waypoint)
                # cv2.imshow('Predict',  aruco_img)
                # cv2.waitKey(0)
                print("localise done")
            time.sleep(3)
        print(f"###################################\nVisited Fruit {waypoint_progress+1}")
    # '''
# 
    # bare minimum waypoint
    # for i in range(8):
        
    # if abs(turn_angle) <=0.8: # ~45deg
        # baseline = 11e-2
    # elif abs(turn_angle) <=1.6: # ~90deg
        # baseline = 10.6e-2
        # 8.135138071066132237e-02
    # elif abs(turn_angle) <=3.2: # ~180deg
        # baseline = 8.3e-2
        # print(baseline)
        # robot_turn(turn_angle=-90*2*np.pi/360,wheel_vel_lin=30,wheel_vel_ang = 20)
        # ppi.set_velocity([0, 0], turning_tick=30, time=1) # stop with delay
        
        
        # baseline =float( input("baseline: "))
        
        # cv2.waitKey(0)
    '''
    waypoints = [[0.4,0],[0.8,0],[0.4,0],[0,0],[-0.4,0],[-0.8,0],[-0.8,0.4],[-0.8,0],[-0.4,0],[0,0],[0.8,0],[0.8,-0.4] ]
    for sub_waypoint in waypoints :
        # Drive to segmented waypoints
        print("    ")
        print("target: "+str(sub_waypoint))
        drive_to_point(sub_waypoint)
        # localize(waypoint)
        # if (i+1)%20 == 0:
        #     print("localising")
        #     localize(sub_waypoint)
        #     # cv2.imshow('Predict',  aruco_img)
        #     # cv2.waitKey(0)
        #     print("localise done")
        # time.sleep(3)
    '''
# """



