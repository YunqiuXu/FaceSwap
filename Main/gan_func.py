#!/usr/bin/env python
# Author: Yunqiu Xu

import cv2
import numpy as np
import tensorflow as tf
from glob import glob
import ImageProcessing


def load_graph(frozen_graph_filename):
    """Load a (frozen) Tensorflow model into memory."""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_filename, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


# Select mouth keypoints --> crop mouth rigion --> resize to 64 * 64 --> drow polylines
# mouth_landmarks: polylines of face landmarks
# mouth_leftup_coord: we have cropped the mouth, this is the upper-left position of cropped image in original image
# m_size: length of edge for cropped image(this is a square)
def getMouthKeypoints(_landmarks,mouth_size = 64): # 
    landmarks = [(_landmarks[0][idx],_landmarks[1][idx]) for idx in range(68)]
    
    border = 5 # border of mouth area
    
    outer_lip = np.array(landmarks[48:60], np.int32).reshape((-1, 1, 2))
    inner_lip = np.array(landmarks[60:68], np.int32).reshape((-1, 1, 2))
    # outer_lip = reshape_for_polyline(landmarks[48:60])  
    # inner_lip = reshape_for_polyline(landmarks[60:68])
    color = (255, 255, 255)
    thickness = 1
    MOUTH_POINTS = list(range(48, 61))
    xs=[]
    ys=[]
    for ind in MOUTH_POINTS:
        xs.append(landmarks[ind][0])
        ys.append(landmarks[ind][1])
                
    xmin,xmax,ymin,ymax = min(xs)-border,max(xs)+border,min(ys)-border,max(ys)+border
    width = xmax-xmin
    length = ymax - ymin
    d = max(width,length)  #length of a edge of square
    centerx = xmin+ width//2    # coordinate of center of the mouth
    centery = ymin + length//2 
    xmin,xmax,ymin,ymax = centerx-d//2,centerx + d//2,centery-d//2,centery+d//2
    black_image = np.zeros((mouth_size,mouth_size), np.uint8)

    ratio = float(mouth_size)/float(d)   #resize ratio
    new_d = mouth_size

    for point in outer_lip:
        x = point[0][0] 
        y = point[0][1]
        resized_x = int(float(x-xmin)/float(d)*new_d)
        resized_y = int(float(y-ymin)/float(d)*new_d)
        point[0][0],point[0][1]= resized_x,resized_y
                
    for point in inner_lip:
        x = point[0][0]
        y = point[0][1]
        resized_x = int(float(x-xmin)/float(d)*new_d)
        resized_y = int(float(y-ymin)/float(d)*new_d)
        point[0][0],point[0][1]= resized_x,resized_y

    cv2.polylines(black_image, [outer_lip], True, color, thickness)  # draw out the lines
    cv2.polylines(black_image, [inner_lip], True, color, thickness)  # draw out the lines

    return black_image,(xmin,ymin),d


# Remap generated mouth back to original size and position
def setMouth(cameraImg, generated_mouth,mouth_leftup_coord,m_size):
    generated_mouth = cv2.resize(generated_mouth,(m_size,m_size))
    xmin,ymin= mouth_leftup_coord[0],mouth_leftup_coord[1]
    mouth = np.zeros((cameraImg.shape[0],cameraImg.shape[1],3), np.uint8) # black image

    for y in range(ymin,ymin+m_size):
        for x in range(xmin,xmin+m_size):
            for j in range(3):
                mouth.itemset((y,x,j),generated_mouth[y-ymin,x-xmin,j]) # put generated mouth to the corresponding position

    return mouth
