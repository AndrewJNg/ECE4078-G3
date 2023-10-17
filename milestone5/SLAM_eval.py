# evaluate the map generated by SLAM against the true map
import ast
import numpy as np
import json
import matplotlib.pyplot as plt
import util.DatasetHandler as dh # save/load functions

def parse_groundtruth(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline()) 
        
        aruco_dict = {}
        for key in gt_dict:
            if key.startswith("aruco"):
                aruco_num = int(key.strip('aruco')[:-2])
                aruco_dict[aruco_num] = np.reshape([gt_dict[key]["x"], gt_dict[key]["y"]], (2,1))
    return aruco_dict

def parse_user_map(fname : str) -> dict:
    with open(fname, 'r') as f:
        try:
            usr_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                usr_dict = ast.literal_eval(f.readline()) 
        aruco_dict = {}
        for (i, tag) in enumerate(usr_dict["taglist"]):
            aruco_dict[tag] = np.reshape([usr_dict["map"][0][i],usr_dict["map"][1][i]], (2,1))
    return aruco_dict

def match_aruco_points(aruco0 : dict, aruco1 : dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue
        
        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
    return keys, np.hstack(points0), np.hstack(points1)

def match_aruco_points_slam(aruco0 : dict):
    points0 = []
    keys = []
    for key in aruco0:
        
        points0.append(aruco0[key])
        keys.append(key)
    return keys, np.hstack(points0)

def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])


    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1/num_points * np.reshape(np.sum(points1, axis=1),(2,-1))
    mu2 = 1/num_points * np.reshape(np.sum(points2, axis=1),(2,-1))
    sig1sq = 1/num_points * np.sum((points1 - mu1)**2.0)
    sig2sq = 1/num_points * np.sum((points2 - mu2)**2.0)
    Sig12 = 1/num_points * (points2-mu2) @ (points1-mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1,-1] = -1
    
    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1,0],R[0,0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert(points.shape[0] == 2)
    
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed =  R @ points + x
    return points_transformed


def compute_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert(points1.shape[0] == 2)
    assert(points1.shape[0] == points2.shape[0])
    assert(points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1-points2).ravel()
    MSE = 1.0/num_points * np.sum(residual**2)

    return np.sqrt(MSE)


def convertArrayToMap(taglist_pred,array):
    # print(taglist_pred)
    map_dict = {}
    for i in taglist_pred:
        if i >15 or i==0:
            continue
        map_dict[f'aruco{i}_0'] = {'x': array[0][i],'y': array[1][i]}
        
    return map_dict


def addFruitToMap(taglist_pred,array):

    # print(taglist_pred)
    map_dict = {}
    for i in taglist_pred:
        # i=-10
        if i ==11:
            map_dict["redapple_0"] = {'x': array[0][i],'y': array[1][i]}
        elif i ==12:
            map_dict["greenapple_0"] = {'x': array[0][i],'y': array[1][i]}
        elif i ==13:
            map_dict["orange_0"] = {'x': array[0][i],'y': array[1][i]}
        elif i ==14:
            map_dict["mango_0"] = {'x': array[0][i],'y': array[1][i]}
        elif i ==15:
            map_dict["capsicum_0"] = {'x': array[0][i],'y': array[1][i]}
        
        
    return map_dict




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    parser.add_argument("groundtruth", type=str, help="The ground truth file name.")
    parser.add_argument("estimate", type=str, help="The estimate file name.")
    args = parser.parse_args()

    # obtain file output
    gt_aruco = parse_groundtruth(args.groundtruth)
    us_aruco = parse_user_map(args.estimate)

    # append point (0,0) as the start origin point
    gt_aruco[0]=np.array([[ 0],[0]])
    us_aruco[0]= np.array([[ 0],[0]])


    taglist, us_vec, gt_vec = match_aruco_points(us_aruco, gt_aruco)
    idx = np.argsort(taglist)
    taglist = np.array(taglist)[idx]
    us_vec = us_vec[:,idx]
    print(us_vec)
    gt_vec = gt_vec[:, idx] 

    # using one known aruco, and known starting point at (0,0), estimate the transform 
    theta, x = solve_umeyama2d(us_vec, gt_vec)
    us_vec_aligned = apply_transform(theta, x, us_vec)

    # calculate error so far
    diff = gt_vec - us_vec_aligned
    rmse = compute_rmse(us_vec, gt_vec)
    rmse_aligned = compute_rmse(us_vec_aligned, gt_vec)

    # apply the same transform to estimated for all points position 
    taglist_pred, us_vec_pred = match_aruco_points_slam(us_aruco)
    idx = np.argsort(taglist_pred)
    taglist_pred = np.array(taglist_pred)[idx]
    us_vec_pred = us_vec_pred[:,idx]
    # print(us_vec)
    us_vec_aligned_pred = apply_transform(theta, x, us_vec_pred)

    print()
    print("The following parameters optimally transform the estimated points to the ground truth.")
    print("Rotation Angle: {}".format(theta))
    print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))
    
    print()
    print("Number of found markers: {}".format(len(taglist)))
    print("RMSE before alignment: {}".format(rmse))
    print("RMSE after alignment:  {}".format(rmse_aligned))
    
    print()
    print('%s %7s %9s %7s %11s %9s %7s' % ('Marker', 'Real x', 'Pred x', 'Δx', 'Real y', 'Pred y', 'Δy'))
    print('-----------------------------------------------------------------')
    
    # '''
    for i in range(len(taglist)):
        print('%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f\n' % (taglist[i], gt_vec[0][i], us_vec_aligned[0][i], diff[0][i], gt_vec[1][i], us_vec_aligned[1][i], diff[1][i]))
    
    ax = plt.gca()
    ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
    ax.scatter(us_vec_aligned_pred[0,:], us_vec_aligned_pred[1,:], marker='x', color='C1', s=100)
    for i in range(len(taglist_pred)):
        # ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist_pred[i], color='C0', size=12)
        ax.text(us_vec_aligned_pred[0,i]+0.05, us_vec_aligned_pred[1,i]+0.05, taglist_pred[i], color='C1', size=12)
    plt.title('Arena')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    plt.legend(['Real','Pred'])
    plt.axis([-1.6,1.6,-1.6,1.6])
    plt.grid()
    plt.show()
    # '''
    with open('lab_output/M5_true_map.txt', 'w') as f:
        json.dump(convertArrayToMap(taglist_pred,us_vec_aligned_pred), f, indent=4)

    with open('lab_output/targets.txt', 'w') as f:
        json.dump(addFruitToMap(taglist_pred,us_vec_aligned_pred), f, indent=4)
        
