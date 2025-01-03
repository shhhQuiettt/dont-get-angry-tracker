from logging import error
import cv2
from cv2.typing import Rect
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

CALIBRATION_FOTOS_DIR = "./fotos"

corner_shape = (6, 8)

objp = np.zeros((corner_shape[0] * corner_shape[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : corner_shape[0], 0 : corner_shape[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

image_names = glob.glob(os.path.join(CALIBRATION_FOTOS_DIR, "*.jpg"))


def calibrate() -> tuple[np.ndarray, np.ndarray, np.ndarray, Rect]:
    """
    returns the camera matrix, distortion coefficients, new camera matrix and region of interest
    then use
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for filename in image_names:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner_shape, None)
        if ret:
            # plt.imshow(
            #     cv2.drawChessboardCorners(img, corner_shape, corners, ret), cmap="gray"
            # )
            # plt.show()
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"Could not find corners in {filename}")
            sys.exit(1)

    gray = cv2.imread(image_names[0], cv2.IMREAD_GRAYSCALE)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    gray = cv2.imread(image_names[7], cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    return mtx, dist, newcameramtx, roi
