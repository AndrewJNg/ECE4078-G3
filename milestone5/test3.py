import numpy as np

def estimate_pose(base_dir, camera_matrix, completed_img_dict, camera_offset):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    target_dimensions = [
        [0.074, 0.074, 0.087],  # Red Apple
        [0.081, 0.081, 0.067],  # Green Apple
        [0.075, 0.075, 0.072],  # Orange
        [0.113, 0.067, 0.058],  # Mango
        [0.073, 0.073, 0.088],  # Capsicum
    ]

    target_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    target_pose_dict = {}
    
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target']
        robot_pose = completed_img_dict[target_num]['robot']
        true_height = target_dimensions[target_num - 1][2]

        pixel_height = box[3][0]
        pixel_center = box[0][0] + box[2][0] / 2
        distance = true_height / pixel_height * focal_length

        image_width = 640
        x_shift = image_width / 2 - pixel_center
        theta = np.arctan(x_shift / focal_length)
        ang = theta + robot_pose[2][0]

        distance_obj = distance / np.cos(theta)
        x_relative = distance_obj * np.cos(theta)
        y_relative = distance_obj * np.sin(theta)

        target_pose = {'x': 0.0, 'y': 0.0}

        # Modify the following lines to account for the camera offset
        camera_x = camera_offset * np.cos(robot_pose[2][0])
        camera_y = camera_offset * np.sin(robot_pose[2][0])

        target_pose['x'] = (x_relative + camera_x) * np.cos(ang) - (y_relative + camera_y) * np.sin(ang)
        target_pose['y'] = (x_relative + camera_x) * np.sin(ang) + (y_relative + camera_y) * np.cos(ang)

        target_pose_dict[target_list[target_num - 1]] = target_pose

    return target_pose_dict