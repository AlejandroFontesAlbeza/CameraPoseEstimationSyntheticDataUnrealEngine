import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def homography(img_intersections, real_world_points):
    number_intersections = sorted(set(img_intersections.keys() & real_world_points.keys()))

    img_pts = np.array([img_intersections[i] for i in number_intersections])
    real_pts = np.array([real_world_points[i] for i in number_intersections])

    if len(number_intersections) >= 4:
        H, _ = cv2.findHomography(real_pts, img_pts)
        return H
    else:
        print("Not enough intersections for homography estimation.")
        return None


def camera_pose_estimation(H, K):
    H = H / np.linalg.norm(H[:,0])
    K_inv = np.linalg.inv(K)

    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]

    r1 = K_inv @ h1
    r2 = K_inv @ h2
    t = K_inv @ h3

    L = 1 / np.linalg.norm(r1)
    r1 *= L
    r2 *= L
    t *= L

    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    R_wc = R.T
    C = -R.T @ t
    cam_position = C.flatten()

    rot = Rot.from_matrix(R_wc)
    rx,ry,rz = rot.as_euler('xyz', degrees = True)
    cam_rotation = np.round(np.array([rx, ry, rz]), 2)
    return cam_position, cam_rotation
