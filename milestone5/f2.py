import cv2
import numpy as np

# Known physical size of the box (in meters)
box_width = 0.06  # Adjust this to match the actual size of the box
box_height = 0.06  # Adjust this to match the actual size of the box


# camera_matrix = np.array([[focal_length_x, 0, principal_point_x],
#                            [0, focal_length_y, principal_point_y],
#                            [0, 0, 1]])  # Replace with your camera intrinsic matrix
# camera_matrix = np.array([[1.07453000e+03, 0, 2.74690405e+02],
#                            [0, 1.07258648e+03, 1.94508578e+02],
#                            [0, 0, 1]])  # Replace with your camera intrinsic matrix
# Camera intrinsic parameters
focal_length_x = 1.07453000e+03  # Focal length in pixels (adjust this to match your camera)
focal_length_y = 1.07258648e+03  # Focal length in pixels (adjust this to match your camera)
principal_point_x = 2.74690405e+02  # Principal point in pixels (adjust this to match your camera)
principal_point_y = 1.94508578e+02  # Principal point in pixels (adjust this to match your camera)

# Bounding box coordinates of the detected box (x_min, y_min, x_max, y_max)

bounding_box = [248, 205, 58, 58]  # Adjust these values based on your detected bounding box
x_min, y_max,width, height = bounding_box
x_max = x_min + width
y_min = y_max - height

bounding_box = [x_min, y_min , x_max, y_max ]  # Adjust these values based on your detected bounding box

# Calculate the center of the bounding box
x_center = (bounding_box[0] + bounding_box[2]) / 2
y_center = (bounding_box[1] + bounding_box[3]) / 2

# Calculate the depth (distance from the camera to the box)
depth = (box_width * focal_length_x) / (bounding_box[2] - bounding_box[0])

# Calculate the X and Y coordinates of the box in the camera's coordinate system
x_camera = (x_center - principal_point_x) * (depth / focal_length_x)
y_camera = (y_center - principal_point_y) * (depth / focal_length_y)

# Assuming the camera is at the origin (0, 0, 0) in the world coordinate system
# The robot's coordinates in the world coordinate system are the same as in the camera's coordinate system
robot_x = x_camera
robot_y = y_camera

print("Robot Coordinates in World Coordinate System:")
print(f"X: {robot_x} meters")
print(f"Y: {robot_y} meters")
print(f"depth: {depth} meters")
