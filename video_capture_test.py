#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 13:24:57 2022

@author: 1517suj
"""

import cv2
# path = 'output_video/lab_test.mp4'
path = 'output_video/marq_1.mp4'
# path = 'output_video/marq_2.mp4'


cap = cv2.VideoCapture(path)
while True:

    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()