from .functions import ransac, compute_homography, filter_and_align_descriptors, extract_features, _get_inlier_count, create_stitched_image, translate_homographies
__authors__ = {'put_your_code_here_ab12game': 'Batman'}



positions=[]
positions2=[]
count=0


import cv2
import re
import os
import random
import numpy as np


logo = cv2.imread("/Users/ankitsharma/Downloads/opencv_logo.png")
cap = cv2.VideoCapture("/Users/ankitsharma/Downloads/still_folder_but_perspective_warp.mp4")

ret, f = cap.read()


def draw_circle(event,x,y,flags,param):
    global positions,count
    # If event is Left Button Click then store the coordinate in the lists, positions and positions2
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(f,(x,y),10,(255,0,0),-1)
        positions.append([x,y])
        if(count!=3):
            positions2.append([x,y])
        elif(count==3):
            positions2.insert(2,[x,y])
        count+=1

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

positions = [[1015, 221], [1552, 340], [846, 578], [1474, 672]]

positions2 = [[1015, 221], [1552, 340], [1474, 672], [846, 578]]


height, width = f.shape[:2]
h1, w1 = logo.shape[:2]

pts1 = np.float32([[0,0],[w1,0],[0,h1],[w1,h1]])
pts2 = np.float32(positions)

h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)


height, width, channels = f.shape
im1Reg = cv2.warpPerspective(logo, h, (width, height))

resized = cv2.resize(im1Reg, (logo.shape[1], logo.shape[0]), interpolation=cv2.INTER_AREA)


mask2 = np.zeros(f.shape, dtype=np.uint8)

roi_corners2 = np.int32(positions2)

channel_count2 = f.shape[2]
ignore_mask_color2 = (255,)*channel_count2

cv2.fillConvexPoly(mask2, roi_corners2, ignore_mask_color2)

mask2 = cv2.bitwise_not(mask2)
masked_image2 = cv2.bitwise_and(f, mask2)

#Using Bitwise or to merge the two images
final = cv2.bitwise_or(im1Reg, masked_image2)
cv2.imwrite('final.png',final)