import CV_eval
import SLAM_eval_ori

import ast
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser("Matching the estimated map and the true map")
# parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
# parser.add_argument("estimate", type=str, help="The estimate file name.")

parser = argparse.ArgumentParser("Final scoring")
parser.add_argument("--true_map", type=str, default='true_map.txt')
parser.add_argument("--map", type=str, default='lab_output\slam.txt')
parser.add_argument("--target_pose", type=str, default='lab_output/targets.txt')
args = parser.parse_args()


target_pose_output = CV_eval.marking(ground_truth=args.true_map, estimation=args.target_pose)
slam_output, taglist, gt_vec, us_vec_aligned,theta,x,rmse,diff = SLAM_eval_ori.slam_marking(ground_truth=args.true_map,estimation=args.map)

print()
print()
print(f"map output: {slam_output}")
print(f"fruit output: {target_pose_output}")


slam_score = (0.3-slam_output)/(0.3-0.01) * 16 + 4
target_pose_score = (0.3-slam_output)/(0.3-0.01) * 16 + 4

print()
print(f"map output: {slam_score}")
print(f"fruit output: {target_pose_score}")

aruco_penalty = 0
fruit_penalty = 1
sum = slam_score + target_pose_score +10 +50 - 5*aruco_penalty - fruit_penalty*2
print(f"final sum: {sum}")
SLAM_eval_ori.print_map(slam_score, taglist, gt_vec, us_vec_aligned,theta,x,rmse,diff)