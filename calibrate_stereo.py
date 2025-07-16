import os
import glob
import cv2
import numpy as np
from pipeline import gstreamer_pipeline  # only import if needed

# --- 1. Calibration parameters ---
BOARD_W, BOARD_H = 9, 6          # Number of inner corners
SQUARE_SIZE = 0.025              # Square size (m)
criteria_subpix = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,    # max iterations
    0.001  # epsilon
)
criteria_stereo = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    100,
    1e-5
)
flags = cv2.CALIB_FIX_INTRINSIC

# 3D real world object points
objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []

# --- 2. Read chessboard corners ---
left_imgs  = sorted(glob.glob("captures/left_*.png"))
right_imgs = sorted(glob.glob("captures/right_*.png"))
for lp, rp in zip(left_imgs, right_imgs):
    imgL = cv2.imread(lp)
    imgR = cv2.imread(rp)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    foundL, cornersL = cv2.findChessboardCorners(grayL, (BOARD_W, BOARD_H), None)
    foundR, cornersR = cv2.findChessboardCorners(grayR, (BOARD_W, BOARD_H), None)
    if not (foundL and foundR):
        continue

    # Refine corners to sub-pixel accuracy
    cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria_subpix)
    cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria_subpix)

    objpoints.append(objp.copy())
    imgpoints_l.append(cornersL)
    imgpoints_r.append(cornersR)

# --- 3. Mono calibration ---
img_size = grayL.shape[::-1]
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, img_size, None, None
)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, img_size, None, None
)
print(f"Left RMS error: {ret_l:.4f}, Right RMS error: {ret_r:.4f}")

# --- 4. Stereo calibration ---
ret_s, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints_l, imgpoints_r,
    mtx_l, dist_l,
    mtx_r, dist_r,
    img_size,
    criteria=criteria_stereo,
    flags=flags
)
print(f"Stereo RMS error: {ret_s:.4f}")

# --- 5. Rectification maps and Q matrix ---
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r,
    img_size, R, T, alpha=0
)

map1_l, map2_l = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2
)
map1_r, map2_r = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2
)

# --- 6. Save parameters ---
os.makedirs("calib", exist_ok=True)
np.savez(
    "calib/stereo_params.npz",
    mtx_l=mtx_l, dist_l=dist_l,
    mtx_r=mtx_r, dist_r=dist_r,
    R1=R1, R2=R2, P1=P1, P2=P2,
    map1_l=map1_l, map2_l=map2_l,
    map1_r=map1_r, map2_r=map2_r,
    Q=Q
)
print("Calibration data saved to calib/stereo_params.npz")
