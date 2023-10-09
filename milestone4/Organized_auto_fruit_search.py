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


"""
def turn_to_fruit(wheel_vel_lin=40,wheel_vel_ang=10):
    img = operate.pibot.pibot.get_image()
    detector_output, img_yolov = yolov.detect_single_image(img)
    timer = time.time()
    Continue =1
    while (Continue == 1):
        try:
            if(len(detector_output)>=1):
                print(detector_output[0][1])
                mean = (detector_output[0][1][0]+detector_output[0][1][2])/2


                while((mean-(640/2)) > 20):
                    lv, rv = operate.pibot.set_velocity([0, -1], turning_tick=wheel_vel_ang, time=0.3)
                    # Have to get drive meas and update slam
                    operate.pibot.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.3)
                    img = operate.pibot.get_image()
                    detector_output, img_yolov = yolov.detect_single_image(img)
                    mean = (detector_output[0][1][0]+detector_output[0][1][2])/2
                    print(mean)
                
                while((mean-(640/2)) < -20):
                    lv, rv = operate.pibot.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=0.3)
                    # Have to get drive meas and update slam
                    operate.pibot.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.3)
                    img = operate.pibot.get_image()
                    detector_output, img_yolov = yolov.detect_single_image(img)
                    mean = (detector_output[0][1][0]+detector_output[0][1][2])/2
                    print(mean)

                operate.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
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
    #     img = self.pibot.get_image()
    #     detector_output, img_yolov = yolov.detect_single_image(img)
    #     lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay


    # while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) > 20):
    #     wheel_vel_lin = 30 # tick to move the robot
    #     wheel_vel_ang = 25
    #     img = self.pibot.get_image()

    #     landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    #     # global robot_pose

        
    #     # turn_angle = -0.2
    #     # # limit angle between -180 to 180 degree (suitable for robot turning)
    #     # turn_angle = (turn_angle) % (2*np.pi) 
    #     # turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

    #     # # make robot turn a certain angle
    #     # turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
    #     # print("Turning for {:.2f} seconds".format(turn_time))

    #     lv, rv = self.pibot.set_velocity([0, -1], turning_tick=wheel_vel_ang)
    #     # self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    # self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    # while((np.mean(aruco_corners[0][0], axis=0)[0]-(640/2)) < -20):
    #     img = self.pibot.get_image()

    #     landmarks, aruco_img = aruco_det.detect_marker_positions(img)
    #     lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    # while((np.mean(input_bbox, axis=0)[0]-(640/2)) < -20):
    #     img = self.pibot.get_image()

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

    #     lv, rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel_ang)
    # self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
"""

'''
def estimate_position():
    img = self.pibot.get_image()
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


################################################################### USER INTERFACE ###################################################################
class Operate:
    def __init__(self):
        # initialise data parameters
        self.pibot = Alphabot(args.ip, args.port)        
        
        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
        self.ekf.robot, marker_length = 0.06) # size of the ARUCO markers
        

        # self.data = None
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        # self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')

        self.scale = 0


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
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(self.drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    def take_and_analyse_picture(self):
        img = self.pibot.get_image()
        # landmarks, aruco_img, boundingbox = aruco_det.detect_marker_positions(img)
        # cv2.imshow('Predict',  aruco_img)
        # cv2.waitKey(0)
        # Remove:
        landmarks, aruco_img = self.aruco_det.detect_marker_positions(img)
        detector_output, img_yolov = yolov.detect_single_image(img)
        print(detector_output)
        return detector_output
    
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
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        self.scale = scale
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

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
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
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

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [2,0]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-2,0]  
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0,2]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0,-2]
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # # save image
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
            #     self.command['save_image'] = True
            # # save SLAM map
            # elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
            #     self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
            else:
                self.command['motion'] = [0,0]
        if self.quit:
            pygame.quit()
            sys.exit()

    def drive_to_point(self, waypoint):
        # rotate robot to turn towards the waypoint
        robot_to_waypoint_angle = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]) # Measured from x-axis (theta=0)
        turn_angle = robot_to_waypoint_angle - robot_pose[2]
        # print(robot_to_waypoint_angle)
        # print(robot_pose[2])
        # print(turn_angle)
        self.robot_turn(turn_angle)
        self.update_slam()
        self.get_robot_pose()
        ####################################################
        # after turning, drive straight to the waypoint
        self.robot_straight(robot_to_waypoint_distance = math.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1]))
        self.update_slam()
        self.get_robot_pose()
        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

    def robot_straight(self, robot_to_waypoint_distance=0,wheel_vel_lin=30,wheel_vel_ang = 25):
        drive_time = (robot_to_waypoint_distance / (self.scale * wheel_vel_lin) )
        # print("Driving for {:.2f} seconds".format(drive_time))

        lv,rv = self.pibot.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
        self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

        self.drive_meas = measure.Drive(lv,rv,drive_time)
    
    def robot_turn(self, turn_angle=0,wheel_vel_lin=30,wheel_vel_ang = 20):
        """
        purpose:
        ### enable the robot to turn a certain angle automatically

        input: 
        ### turn_angle: angle to rotate on the spot radians (negative value means turn right)

        return:
        ### drive_meas
        """
        # wheel_vel_lin = 30 # tick to move the robot
        # wheel_vel_ang = 10
        global baseline
        if abs(turn_angle) <=1.6: # ~90deg
            baseline = 10.6e-2

        elif abs(turn_angle) <=3.2: # ~180deg
            baseline = 8.9e-2

        
        # limit angle between -180 to 180 degree (suitable for robot turning)
        turn_angle = (turn_angle) % (2*np.pi) 
        turn_angle = turn_angle-2*np.pi if turn_angle>np.pi else turn_angle

        # make robot turn a certain angle
        turn_time = (abs(turn_angle) * ((baseline/2)/(self.scale*wheel_vel_ang))).tolist()
        # print("Turning for {:.2f} seconds".format(turn_time))

        lv, rv = self.pibot.set_velocity([0, np.sign(turn_angle)], turning_tick=wheel_vel_ang, time=turn_time)
        self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
        
        self.drive_meas = measure.Drive(lv,rv,turn_time)
        
    
    def get_robot_pose(self):
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
        ## method 2: Using SLAM through EKF to get robot_pos
        robot_pose = self.ekf.robot.state.reshape(-1)
        print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")

    
    def localize(self, waypoint): # turn and call get_robot_pose
        global robot_pose
        # baseline = 11.6e-2
        turn_angle = 2*np.pi/10
        wheel_vel_ang = 10
        turn_time = turn_angle * ((baseline/2)/(self.scale*wheel_vel_ang))
        robot_pose_previous = robot_pose
        print("\nLocalising Now")
        turn_count = 0
        latest_pose = np.zeros((0,3))
        aruco_3_skip_flag = 0
        while turn_count<10:
            lv,rv = self.pibot.set_velocity([0, 1], turning_tick=wheel_vel_ang, time=turn_time)
            turning_drive_meas = measure.Drive(lv,rv,turn_time)
            self.pibot.set_velocity([0, 0], turning_tick=wheel_vel_ang, time=0.8) # immediate stop with small delay
            robot_pose, lms = self.get_robot_pose(turning_drive_meas)
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

####################################################################################################
# main loop
if __name__ == "__main__":
    # arguments for starting command
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.209')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")

    # For debuging
    parser.add_argument("--localize", type=int, default=0)
    parser.add_argument("--wait", type=int, default=0)
    # parser.add_argument("--save_data", action='store_true') # Not implemented
    # parser.add_argument("--play_data", action='store_true') # Not implemented
    args, _ = parser.parse_known_args()

    # file output location for predicted output by neural network
    file_output = None
    output = dh.OutputWriter('lab_output')
    pred_fname = ''

    # neural network file location
    args.ckpt = "network/scripts/model/yolov8_model_best.pt"
    yolov = Detector(args.ckpt)

    operate = Operate(args)
    operate.init_ekf(args.calib_dir, args.ip)

    # Initiate UI
    initiate_UI()
    operate = Operate()

    ####################################################################################################
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    
    landmarks = []
    print(aruco_true_pos)
    
    for i,landmark in enumerate(aruco_true_pos):
        measurement_landmark = measure.Marker(np.array([[landmark[0]],[landmark[1]]]),i+1)
        landmarks.append(measurement_landmark)
    print(landmarks)
    operate.ekf.add_landmarks(landmarks)

    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    ####################################################################################################

    global robot_pose
    robot_pose = [0.,0.,0.]
    operate.pibot.set_velocity([0, 0]) # stop with delay

    ####################################################################################################

    '''
    while True:
        try:
            print()
            print("next")
            x = input("input, x: ")
            y = input("input, y: ")
            sub_waypoint = [float(x),float(y)]
            operate.drive_to_point(sub_waypoint)
        except:
            print("enter again")
    '''


    ########################################   A* CODE INTEGRATED ##################################################
    global waypoints
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
            operate.drive_to_point(sub_waypoint)
            print("after_POSE",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)
        
            if args.wait:
                cv2.imshow('Predict', operate.aruco_img)
                cv2.waitKey(0)
            if args.localize:
                if (i+1)%3 == 0:
                    print("localising")
                    operate.localize(sub_waypoint)
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
        # self.pibot.set_velocity([0, 0], turning_tick=30, time=1) # stop with delay
        
        
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



