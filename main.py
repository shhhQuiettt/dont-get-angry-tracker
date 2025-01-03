import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from calibration import calibrate

video_path = "./videos/gameplay.mkv"

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit(1)

mtx, dist, newcameramtx, (x, y, w, h) = calibrate()
vid_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)[
        y : y + h, x : x + w
    ]
    plt.imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
    plt.show()
    vid_writer.write(undistorted_frame)

cap.release()
vid_writer.release()
# fig, axs = plt.subplots(1, 2)
# axs[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# axs[0].set_title("Original Frame")
# axs[0].axis("off")
# axs[1].imshow(cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB))
# axs[1].set_title("Undistorted Frame")
# axs[1].axis("off")
# plt.show()
