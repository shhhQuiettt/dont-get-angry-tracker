import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from calibration import calibrate

vv.set(cv2.CAP_PROP_POS_FRAMES, 200)
while vv.isOpened():
    ret, frame = vv.read()

    if ret:

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_frame, 225, 255, cv2.THRESH_BINARY)
        # imshow(thresh)
        kernel = np.ones((5, 5), np.uint8)
        # first colsing
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # imshow(closing)
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=3)
        # imshow(opening)
        op_frame = opening.copy()

        _pleng = op_frame.copy()
        # Apply template Matching
        res = cv2.matchTemplate(_pleng, pleng_template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if cv2.TM_CCOEFF in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

        # crop the detected cube
        cube = frame[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

        # results_rgb.append(_pleng)
        # results_prob.append((res - res.min()) * 255 / (res.max() - res.min()))

        free_kick_track.write(cube)
    else:
        break

free_kick_track.release()
