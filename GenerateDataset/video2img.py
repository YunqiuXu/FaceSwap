#!/usr/bin/env python

# Read all videos, get video capture per 5 frames
# you need to make a folder "img_dataset"
# The name of video is like 01.mp4, 02.mp4 ...
# The output image size is 600 * 720

# How to run it : python video2img.py


import cv2
import skvideo.io
import sys
from glob import glob

video_names = sorted(glob("*.mp4"))
for video_name in video_names:

    # read current video
    video = skvideo.io.vreader(video_name)
    print("Current video : " + str(video_name))
    count = 1
    img_count = 1
    
    for frame in video:
        # cvt to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # make crop to get smaller image
        height, width = frame.shape[:2]
        start = (width - height) / 2
        end = (width + height) / 2
        frame_crop = frame[:600, start:end, :]
        
        # show frames
        cv2.imshow("original frame", frame)
        cv2.imshow("cropped frame", frame_crop)
        print frame_crop.shape

        if count % 5 == 0:
            # set image name
            head = "img_dataset/" + str(video_name[:-4])
            zeros = '0' * (5 - len(str(img_count)))
            img_name = head + zeros + str(img_count) + ".jpg"
            img_count += 1
            
            # save img
            cv2.imwrite(img_name, frame_crop)

        count += 1

        if cv2.waitKey(1) & 0xff == ord('q'):
                break
    
    print(str(video_name) + " is finished!")
    print("-----")

cv2.destroyAllWindows()


