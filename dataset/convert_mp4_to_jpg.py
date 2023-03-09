import cv2
import os
import numpy as np
import glob

data_root = '/home/boweiren/Workspace/anomly_feature.pytorch/Anomaly-Videos'
video_paths = glob.glob(os.path.join(data_root, '*/*'))

for p in video_paths:

    tmp = p.split('/')
    tmp[-3] += '-frames'
    tmp[-1] = tmp[-1].split('.')[0].replace('_x264', '')
    os.makedirs('/'.join(tmp))

    cap = cv2.VideoCapture(p)

    # Initialize a counter variable for the image filenames
    count = 0

    # Loop through each frame of the video
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Save the frame as a JPEG image
        filename = '/'.join(tmp) + '/' + format(count, '04d') + '.jpg'
        cv2.imwrite(filename, frame)

        # Increment the counter
        count += 1

    # Release the video capture object
    cap.release()

    print('{count} images saved.'.format(count=count))

