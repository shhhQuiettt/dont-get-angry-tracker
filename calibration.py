from logging import error
import cv2
from cv2.typing import Rect
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

CALIBRATION_FOTOS_DIR = "./fotos/calibration_phone/"

corner_shape = (4, 9)


def calibrate() -> tuple[np.ndarray, np.ndarray, np.ndarray, Rect]:
    objp = np.zeros((corner_shape[0] * corner_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : corner_shape[0], 0 : corner_shape[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    image_names = glob.glob(os.path.join(CALIBRATION_FOTOS_DIR, "*"))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for filename in image_names:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner_shape, None)
        if ret:
            # plt.imshow(
            #     cv2.drawChessboardCorners(img, corner_shape, corners, ret), cmap="gray"
            # )
            # plt.title(filename)
            # plt.show()
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            raise ValueError(f"Could not find corners in {filename}")

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, gray.shape[::-1], alpha=1
        )
    except UnboundLocalError:
        raise ValueError("No files found")

    if not ret:
        raise ValueError("Could not calibrate the camera")

    return mtx, dist, new_camera_mtx, roi


calibrate()
