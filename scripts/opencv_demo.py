#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Created by eh at 2020/5/25

import cv2
import numpy as np

cap = cv2.VideoCapture("../data/20200522_114024_974_797.mp4")

# while(1):
#     # get a frame
#     ret, frame = cap.read()
#     # show a frame
#     cv2.imshow("capture", frame)
#     if cv2.waitKey(100) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

fps = cap.get(cv2.CAP_PROP_FPS)
ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(ori_w)
ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_saver = cv2.VideoWriter('1.mp4', fourcc, fps, (ori_h,ori_w))

while cap.isOpened():
    ret, frame = cap.read()
    # frame = np.rot90(frame, -1)
    trans_img = cv2.transpose(frame)
    frame = cv2.flip(trans_img, 1)

    video_saver.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

