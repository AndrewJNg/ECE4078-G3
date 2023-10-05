
# import numpy as np
# waypoint = [0,0,0]
# robot_pose=[0,0,0]

# print( np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0]))

import numpy as np

def image_to_camera_coordinates(bounding_box, camera_matrix, rotation_matrix, translation_vector):
    # Define the 2D bounding box points
    x_min, y_min, x_max, y_max = bounding_box

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


# Example usage
'''
bounding_box = [100, 200, 300, 400]  # Replace with your bounding box coordinates (x_min, y_min, x_max, y_max)
# camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
#                            [0, focal_length_y, principal_point_y],
#                            [0, 0, 1]])  # Replace with your camera intrinsic matrix
rotation_matrix = np.array([[r11, r12, r13],
                            [r21, r22, r23],
                            [r31, r32, r33]])  # Replace with your camera rotation matrix

translation_vector = np.array([tx, ty, tz])  # Replace with your camera translation vector
'''

fileK = "calibration/param/intrinsic.txt"
camera_matrix = np.loadtxt(fileK, delimiter=',')
# camera_matrix = np.array([[1.07453000e+03, 0, 2.74690405e+02],
#                            [0, 1.07258648e+03, 1.94508578e+02],
#                            [0, 0, 1]])  # Replace with your camera intrinsic matrix

rotation_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])  # Replace with your camera rotation matrix

translation_vector = np.array([0, 0, 0])  # Replace with your camera translation vector

bounding_box = [0,0,640,480]
point_3d = image_to_camera_coordinates(bounding_box, camera_matrix, rotation_matrix, translation_vector)
print("3D Point in Camera Coordinates:", point_3d)
