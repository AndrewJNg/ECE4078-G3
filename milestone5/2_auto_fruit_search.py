# M4 - Autonomous fruit searching
# New: Update pathfinding

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import ast
import argparse
import time
import math
import fruit_est as detect
import generateWaypoints as wp

# import neural network detector
from network.scripts.detector import Detector # modified for yolov8

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco
import util.DatasetHandler as dh # save/load functions
import shutil # python package for file operations
import SLAM_eval

# import utility functions
sys.path.insert(0, "util")
from util.pibot import Alphabot
import util.measure as measure
import pathFind
import pygame

import fruit_detector

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
        self.clock = pygame.time.Clock()
        self.clock.tick(30)
        self.last_time=0

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

        time_remain = time.time() - self.start_time
        if (time.time()-self.last_time)>10:
            self.last_time= time.time()
            print(f"curr_time: {round((time_remain) / 60)} minute {round(time_remain%60)} seconds")
            # print(f"Current time: {round(time_remain,2)}")
        # time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Up: {time_remain:03.0f}s'
        # elif int(time_remain)%2 == 0:
        #     time_remain = "Time Is Up !!!"
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
    pygame.display.set_caption('ECE4078 2023 Lab')
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

################################################################### Map functions ###################################################################
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
            x = gt_dict[key]['x']
            y = gt_dict[key]['y']

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

################################################################### Helper functions ###################################################################
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

def angleToPulse(angle):

    ## servo calibration
    # while True:
    #     try:
    #         print()
    #         print("next")
    #         x = input("input, pulse: ")
    #         ppi.set_servo(int(x))

    #     except:
    #         if(str(x)=='z'):
    #             break
    #         print("enter again")

    #calibration
    xp= [-90*np.pi/180,-45*np.pi/180,0*np.pi/180,45*np.pi/180,90*np.pi/180]
    yp= [570,1100,1650,2150,2700]
    
    pulse = int(np.interp(angle,xp,yp))
    # print(pulse)
    return pulse

################################################################### Robot drive functions ###################################################################
def drive_to_point(waypoint):
    global robot_pose
    
    ####################################################
    # rotate robot to turn towards the waypoint
    robot_to_waypoint_angle = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]) # Measured from x-axis (theta=0)
    turn_angle = robot_to_waypoint_angle - robot_pose[2]
    robot_turn(turn_angle)
    operate.draw(canvas)
    pygame.display.update()

    ####################################################
    # after turning, drive straight to the waypoint
    distance = math.hypot(waypoint[0]-robot_pose[0], waypoint[1]-robot_pose[1])
    robot_straight(robot_to_waypoint_distance = distance)
    operate.draw(canvas)
    pygame.display.update()
    # print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

def robot_turn(turn_angle=0,wheel_vel_lin=30,wheel_vel_ang = 20):
    """
    purpose:
    ### enable the robot to turn a certain angle automatically
   

    input: 
    ### turn_angle: angle to rotate on the spot radians (negative value means turn right)

    output:
    ### physical robot: robot would rotate by turn_angle amount
    """
    global baseline
    if abs(turn_angle) <=np.deg2rad(40): # <=30deg
        baseline = 15.5e-2
    elif abs(turn_angle) <=np.deg2rad(50): # ~45deg
        baseline = 11.2e-2
    elif abs(turn_angle) <=np.deg2rad(100): # ~90deg
        baseline = 10.2e-2
    elif abs(turn_angle) <=np.deg2rad(145): # ~135deg
        baseline = 9.5e-2
    else:  # ~180deg
        baseline = 9.5e-2 
    # print(f"baseline: {baseline}")
    
    # if abs(turn_angle) <=0.8: # <45deg
    #     baseline = 15.5e-2
    # elif abs(turn_angle) <=1.6: # ~90deg
    #     baseline = 11.2e-2
    # elif abs(turn_angle) <=3.2: # ~180deg
    #     baseline = 9.0e-2
    '''
    #base values
    if abs(turn_angle) <=0.8: # <45deg
        baseline = 14.5e-2
    elif abs(turn_angle) <=1.6: # ~90deg
        baseline = 10.2e-2
    elif abs(turn_angle) <=3.2: # ~180deg
        baseline = 9.0e-2
    '''

    # make robot turn a certain angle
    turn_angle = clamp_angle(turn_angle) # limit angle between -180 to 180 degree (suitable for robot turning)
    turn_time = (abs(turn_angle) * ((baseline/2)/(scale*wheel_vel_ang))).tolist()
    lv, rv = ppi.set_velocity([0, np.sign(turn_angle)], turning_tick=wheel_vel_ang, time=turn_time)

    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay
    
    drive_meas = measure.Drive(lv,rv,turn_time)
    get_robot_pose(drive_meas)

def robot_straight(robot_to_waypoint_distance=0, wheel_vel_lin=30, wheel_vel_ang = 25):
    drive_time = (robot_to_waypoint_distance / (scale * wheel_vel_lin) )
    # if drive_time >3:
        # drive_time =3
    # print("Driving for {:.2f} seconds".format(drive_time))

    lv,rv = ppi.set_velocity([1, 0], tick=wheel_vel_lin, time=drive_time)
    ppi.set_velocity([0, 0], turning_tick=wheel_vel_lin, time=0.5) # stop with delay

    drive_meas = measure.Drive(lv,rv,drive_time)
    get_robot_pose(drive_meas)

################################################################### Pictures and model ###################################################################
def take_and_analyse_picture():
    global aruco_img

    global camera_matrix
    global dist_coeffs
    marker_pose = None
    # aruco_id =[]

    img = ppi.get_image()
    landmarks_aruco, aruco_img, boundingbox, aruco_id = aruco_det.detect_marker_positions(img)

    # visualise
    operate.draw(canvas)
    pygame.display.update()

    landmarks_fruits, fruit_img = fruit_detector.detect_fruit_landmark(yolov=yolov,img=img,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs)

    # visualise
    operate.draw(canvas)
    pygame.display.update()


    landmarks_combined = []
    landmarks_combined.extend(landmarks_aruco)
    landmarks_combined.extend(landmarks_fruits)
    
    # print(f"1_ids : {aruco_id}")

    return landmarks_combined, marker_pose, boundingbox, aruco_id
    # return landmarks, detector_output,aruco_corners

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

################################################################### SLAM - EKF method  ###################################################################
def get_robot_pose(drive_meas,servo_theta=0):
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
    ## method 2: Using SLAM through EKF filtering
    # '''
    global robot_pose
    global landmarks # NEW Added
    landmarks, marker_pose, boundingbox, aruco_id = take_and_analyse_picture()
    ekf.predict(drive_meas,servo_theta=servo_theta)
    ekf.add_landmarks(landmarks)
    ekf.update(landmarks)

    robot_pose = ekf.robot.state.reshape(-1)
    # print(f"Get Robot pose : [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")

    # visualise
    operate.draw(canvas)
    pygame.display.update()

    return robot_pose, landmarks

def localize(increment_angle = 5): # turn and call get_robot_pose
    global robot_pose
    global landmarks
    
    landmark_counter = 0
    lv,rv=ppi.set_velocity([0, 0], turning_tick=30, time=0.8) # immediate stop with small delay
    drive_meas = measure.Drive(lv,rv,0.8,left_cov=0.00001,right_cov=0.00001)

    # look right first
    ppi.set_servo(angleToPulse(-90*np.pi/180))
    time.sleep(0.3)
    get_robot_pose(drive_meas,servo_theta=-90*np.pi/180)
    time.sleep(0.2)
    
    # increment by a small angle until it finish 180 degree
    
    current_angle = -90
    for i in range(int(180/increment_angle)):
        current_angle+=increment_angle
        ppi.set_servo(angleToPulse(current_angle*np.pi/180))
        time.sleep(0.4)
        _, landmarks = get_robot_pose(drive_meas,servo_theta=increment_angle*np.pi/180)
        time.sleep(0.2)
        landmark_counter+=len(landmarks)

    # look back at center
    ppi.set_servo(angleToPulse(0*np.pi/180))
    time.sleep(0.3)
    get_robot_pose(drive_meas,servo_theta=-90*np.pi/180)
    time.sleep(0.5)
    # print(f"Landmarks: {landmark_counter}")
    return landmark_counter

def validify_base_aruco(aruco_id,boundingbox,current_angle,aruco_target_pose):
    global waypoint_arr
    
    if len(aruco_id) ==1:
        # print(boundingbox[0][0]-320)
        if(abs(boundingbox[0][0]-320)<160):
            x,y= take_marker_pose(boundingbox,robot_pose)
            # x = np.around(x,1)
            # y = np.around(y,1)
            aruco_target_pose[f'aruco{int(aruco_id[0][0])}_0'] ={'x': x,'y': y}
            print(f"base_aruco: {int(aruco_id[0][0])} at [{x},{y}]")
            
            if(current_angle == -90):
                waypoint_arr.append([x,y+0.4])
            elif(current_angle == 0):
                waypoint_arr.append([x-0.4,y])
            elif(current_angle == 90):
                waypoint_arr.append([x,y-0.4])

    elif len(aruco_id) >=2:
        # print(boundingbox)
        
        x_values = np.array([item[0] for item in boundingbox])
        # print(f"aruco_id: {aruco_id} bbox: {boundingbox}, x: {x_values}")
        offset = abs(320*np.ones_like(x_values) - x_values)
        # print(f"offset: {offset}")
        index = np.argmin(offset)
        # print(f"min: {index}")
        id = aruco_id[index]
        # print(f"id: {id}")
        
        if(abs(boundingbox[index][0]-320)<160):
            x,y= take_marker_pose(boundingbox,robot_pose)
            # x = np.around(x,1)
            # y = np.around(y,1)
            aruco_target_pose[f'aruco{int(aruco_id[0][0])}_0'] ={'x': x,'y': y}
            print(f"base_aruco: {id} at [{x},{y}]")
            
            if(current_angle == -90):
                waypoint_arr.append([x,y+0.4])
            elif(current_angle == 0):
                waypoint_arr.append([x-0.4,y])
            elif(current_angle == 90):
                waypoint_arr.append([x,y-0.4])

def generateBaseMap(fname):
    aruco_target_pose = {}
    increment_angle = 45
    current_angle = -90

    # look right first
    lv,rv=ppi.set_velocity([0, 0], turning_tick=30, time=0.8) # immediate stop with small delay
    drive_meas = measure.Drive(lv,rv,0.8,left_cov=0.00001,right_cov=0.00001)
    ppi.set_servo(angleToPulse(-90*np.pi/180))
    time.sleep(0.3)

    # Get picture and estimate marker position
    get_robot_pose(drive_meas,servo_theta=-90*np.pi/180)
    landmarks_combined, current_marker_pose, boundingbox, aruco_id = take_and_analyse_picture()
    validify_base_aruco(aruco_id,boundingbox,current_angle,aruco_target_pose)
    time.sleep(0.2)

    
    for i in range(int(180/increment_angle)):
        current_angle+=increment_angle
        ppi.set_servo(angleToPulse(current_angle*np.pi/180))
        time.sleep(0.4)
        _, landmarks = get_robot_pose(drive_meas,servo_theta=increment_angle*np.pi/180)
        # Get picture and estimate marker position
        landmarks_combined, current_marker_pose, boundingbox, aruco_id = take_and_analyse_picture()
        validify_base_aruco(aruco_id,boundingbox,current_angle,aruco_target_pose)
        time.sleep(0.2)

    with open('lab_output/base_map.txt', 'w') as f:
        json.dump(aruco_target_pose, f, indent=4)
        
    print(f"######################################################################")
    print(f"Base arucos: {aruco_target_pose}")
    print(f"######################################################################")

    # look back at center
    ppi.set_servo(angleToPulse(0*np.pi/180))
    time.sleep(0.5)
    get_robot_pose(drive_meas,servo_theta=-90*np.pi/180)
    time.sleep(0.2)
    return None

def take_marker_pose(box,robot_pose):
    global camera_matrix
    true_height = 0.06
    focal_length = camera_matrix[0][0]
    camera_offset = 0.110 #3.5cm
    
    ## ASSUMING ONLY 1 BOX
    distance = focal_length * true_height/box[0][3]

    x = robot_pose[0] + np.cos(robot_pose[2]) * (distance + camera_offset)
    y = robot_pose[1] + np.sin(robot_pose[2]) * (distance + camera_offset) 
    world_frame_pos = [x,y]

    return world_frame_pos


def getMinTurnPath(available_waypoints_with_dist, current_start_pos):
    # Initializing array for paths and turns for each waypoints
    temp_paths = [50 for z in range(len(available_waypoints_with_dist))]
    turn_arr = [50 for z in range(len(available_waypoints_with_dist))]
    # Loop through each waypoints
    print(f"available_waypoints_with_dist {available_waypoints_with_dist}")
    for i, (pose, dist) in enumerate(available_waypoints_with_dist):
        temp_path, turns = pathFind.main(current_start_pos, pose, fruits_true_pos)
        if temp_path == []:
            print("Path empty")
            temp_paths[i] = [[]]
        else: 
            temp_paths[i] = temp_path
        turn_arr[i] = turns
        print(" ")
    # Find index of least turn path
    min_turn = 50
    min_turn_dist = 50
    index_min_turn = -1
    for j, turns in enumerate(turn_arr):
        if turns < min_turn:
            min_turn = turns
            min_turn_dist = dist
            index_min_turn = j
        elif turns == min_turn: # If number of turns is equal
            if dist < min_turn_dist: # If distance is lesser, update path to be best path
                min_turn_dist = dist
                index_min_turn = j

    # Assign current path and waypoint using index found
    path = temp_paths[index_min_turn]
    # Pop first position
    # path.pop(0)
    # waypoints.append(available_waypoints_with_dist[index_min_turn][0])
    
    return path, min_turn

def update_true_map():
    global fruits_list
    global fruits_true_pos
    global aruco_true_pos
    
    output_path.write_map(ekf)
    SLAM_eval.generate_map(base_file=args.base_map,slam_file='lab_output/slam.txt')
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
################################################################### Main  ###################################################################
# main loop
if __name__ == "__main__":
    ## Robot connection setup
    # arguments for starting command
    parser = argparse.ArgumentParser("Fruit searching")
    
    parser.add_argument("--base_map", type=str, default='lab_output/base_map.txt')
    parser.add_argument("--map", type=str, default='lab_output\M5_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.137.156')
    parser.add_argument("--port", metavar='', type=int, default=8000)
    parser.add_argument("--yolo", metavar='', type=int, default=0)
    parser.add_argument("--ckpt", metavar='', type=str, default='network/scripts/model/yolov8_model_best.pt')
    args, _ = parser.parse_known_args()
    

    # robot startup with given variables
    ppi = Alphabot(args.ip,args.port)
    
    # file output location for predicted output by neural network
    file_output = None
    output = dh.OutputWriter('lab_output')
    pred_fname = ''

####################################################
    ## Obtain robot settings for camera and motor (found by calibration)
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

    global output_path
    output_path = dh.OutputWriter('lab_output')
    # neural network file location
    global yolov
    yolov = Detector(args.ckpt)

####################################################
# Initiate UI
    start = initiate_UI()
    operate = Operate()

####################################################
    global robot_pose
    robot_pose = [0.,0.,0.]
    current_start_pos = [robot_pose[0],robot_pose[1]]
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    aruco_det = aruco.aruco_detector(robot, marker_length = 0.06)
    ekf = EKF(robot)
    ppi.set_servo(angleToPulse(0*np.pi/180))
    # while True:
    #     # robot_straight(0.8)
    #     robot_turn(turn_angle=np.deg2rad(180))
    #     x = input()
    #     pass
####################################################
    # Generate Aruco base map
    global waypoint_arr 
    # waypoint_arr = [[-0.8,0]]
    waypoint_arr = []
    generateBaseMap("lab_output/base_map.txt")
    search_list = read_search_list()


####################################################
    ## Set up all EKF using given values in true map
    output_path.write_map(ekf)
    SLAM_eval.generate_map(base_file=args.base_map,slam_file='lab_output/slam.txt')
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
########################################   A* CODE INTEGRATED ##################################################
    
    #### Start Localizing on Origin ####
    localize(15)
    
    # '''
    # pose_arr = [[-0.8,0], [-1.2,1.2],[1.2,1.2],[1.2,-1.2],[-1.2,-1.2]]
    # waypoint_arr = [[0,0.4],[0,-0.4]]
    # waypoint_arr = [[-0.8,0], [0.8,0]]
    print(f"waypoint_arr: {waypoint_arr}")
    for waypoint in waypoint_arr:
        #### Localizing After Fruit Visit ####
        robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
        localize(15)
        
        while current_start_pos != waypoint:
            ## Update ##
            # Update EKF Outputs
            update_true_map()
            # Update path
            if current_start_pos == waypoint:
                print(f"Error: current pos = waypoint {current_start_pos}")
            path, turns = pathFind.main(current_start_pos, waypoint, fruits_true_pos)
            if not path:
                print("Path empty :( Me will crash")
                # dist = 0.2
                # available_waypoints_with_dist = wp.generateWaypoints() 
                # path, _ = getMinTurnPath(available_waypoints_with_dist, current_start_pos)
            if current_start_pos == path[0]:
                path.pop(0)
            target_pose = path[0]
            print(f'Last pose: {current_start_pos}')
            print("Target: "+str(target_pose))
            print(f'Path: {path}')
            print(f'Turns for path: {turns}')

            # Drive to segmented waypoints
            # operate.draw(canvas)
            pygame.display.update() 
            drive_to_point(target_pose)
            print(f'Current pose: {current_start_pos}')
            print("Slam Pose: ", round(robot_pose[0],2), round(robot_pose[1],2), round(robot_pose[2]*180/np.pi,2))
            print("    ")
            # Update start pos
            current_start_pos = target_pose
            
            landmark_counter = localize(15)
            print("Slam Pose After Localize: ", round(robot_pose[0],2), round(robot_pose[1],2), round(robot_pose[2]*180/np.pi,2))
            if landmark_counter == 0: # If seen markers not more than 2
                print("Seen 0 landmarks. Localize agian")
                # Turn 180 deg and localize
                robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
                localize(15)
        print(f"\n### Reached {waypoint} ###\n")
            
    
    output_path.write_map(ekf)
    SLAM_eval.generate_map(base_file=args.base_map,slam_file='lab_output/slam.txt')
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    # '''
    
    
    '''
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()

    #### AUTONOMOUS NAVIGATION ####
    # Get updated waypoints
    waypoints_compiled = wp.generateWaypoints(robot_pose, search_list, fruits_list, fruits_true_pos)
    available_waypoints_with_dist = waypoints_compiled[0]
    # Get updated path
    path, min_turn = getMinTurnPath(available_waypoints_with_dist, current_start_pos)
    path.pop(0)
    target_pose = path[0]
    # Get updated waypoint
    waypoint = path[-1]
    # Print path
    print(f'Path: {path}')
    print(f'Turns left: {min_turn}')
    # waypoints_compiled = wp.generateWaypoints(robot_pose, search_list, fruits_list, fruits_true_pos)
    for fruit_progress in range(len(search_list)):
        #### Localizing After Fruit Visit ####
        robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
        localize(30)
        
        while current_start_pos != waypoint:
            ## Update ##
            # Update EKF Outputs
            output_path.write_map(ekf)
            SLAM_eval.generate_map(base_file=args.base_map,slam_file='lab_output/slam.txt')
            fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
            search_list = read_search_list()

            print("After_slam_coord: ",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)
            # Get updated fruit pos & obstacle pos
            # Get updated waypoints
            waypoints_compiled = wp.generateWaypoints(robot_pose, search_list, fruits_list, fruits_true_pos)
            available_waypoints_with_dist = waypoints_compiled[fruit_progress]
            # Get updated path
            path, min_turn = getMinTurnPath(available_waypoints_with_dist, current_start_pos)
            print(f"Path b4 popping")
            if current_start_pos == path[0]:
                path.pop(0)
            target_pose = path[0]
            # Get updated waypoint
            waypoint = path[-1]
            # Print path
            print(f'Path: {path}')
            print(f'Turns left: {min_turn}')

            # Drive to segmented waypoints
            # operate.draw(canvas)
            pygame.display.update()
            print("    ")
            print("Target: "+str(target_pose))
            drive_to_point(target_pose)
            print("Current_coord_pose",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)
            # Update start pos
            current_start_pos = target_pose
            
            landmark_counter = localize(30)
            if landmark_counter == 0: # If seen markers not more than 2
                print(f"Seen 0 landmarks during localize ({landmark_counter}). Localize agian")
                # Turn 180 deg and localize
                robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
                localize(30)
            
        
        output_path.write_map(ekf)
        
        print(f"######################################################################")
        print(f"Visited Fruit {fruit_progress+1}: {search_list[fruit_progress]} at {current_start_pos}")
        print(f"######################################################################")
        ppi.set_velocity([0, 0], turning_tick=0, time=3) # stop with delay
    pygame.quit()
    sys.exit()
    '''
    # """
    for fruit_progress in range(len(search_list)):
        update_true_map()

        waypoints_compiled = wp.generateWaypoints(robot_pose, search_list, fruits_list, fruits_true_pos)
        available_waypoints_with_dist = waypoints_compiled[fruit_progress]

        # Get initial path
        path, min_turn = getMinTurnPath(available_waypoints_with_dist, current_start_pos)
        print(f'Initial Path: {path}')
        print(f'Turns for path: {min_turn}')

        #### Start Localizing on Origin ####
        robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
        localize(15)

        #### Main Algorithm ####
        for i, target_pose in enumerate(path, 3):
            # Drive to segmented waypoints
            # operate.draw(canvas)
            pygame.display.update()
            print("    ")
            print(f'Current pose: {current_start_pos}')
            print("Target: "+str(target_pose))
            drive_to_point(target_pose)
            current_start_pos = target_pose
            print("Slam pose", round(robot_pose[0],2), round(robot_pose[1], 2), round(robot_pose[2]*180/np.pi,2))
            print("Target: "+str(target_pose))
            landmark_counter = localize(30)
            if landmark_counter == 0: # If seen markers not more than 2
                print(f"Seen 0 landmarks during localize ({landmark_counter}). Localize agian")
                # Turn 180 deg and localize
                robot_turn(turn_angle=180*np.pi/180,wheel_vel_lin=30,wheel_vel_ang = 20)
                localize(15)
            print("Pose after localizing",robot_pose[0],robot_pose[1],robot_pose[2]*180/np.pi)

            ## Update Positions and Target Waypoints##
            # Get updated fruit pos & obstacle pos
            update_true_map()
            
        print(f"######################################################################")
        print(f"Visited Fruit {fruit_progress+1}: {search_list[fruit_progress]} at {current_start_pos}")
        print(f"######################################################################")
        # ppi.set_velocity([0, 0], turning_tick=0, time=3) # stop with delay


    pygame.quit()
    sys.exit()
    # """