import time
import numpy as np
import pickle
import os
from scipy.signal import savgol_filter
import cv2
import socket
import csv


class PuckTracker():
    def __init__(self):
        self.start_time = time.time()
        self.dir_path = os.path.dirname(os.path.realpath(__file__))  # directory of this python file
        
        self.puck_pos = [-100, -100]
        self.puck_vel = [0.0, 0.0]
        
        # Make the image 200x400, two pixels per cm
        self.des_image_shape = [200, 400]
        self.pixels_to_cm = 0.5
        # Use get_corners.py to get from_corners
        self.from_corners = [[11,71],[604,57],[633,332],[13,368]]
        self.to_corners = [[0,400],[0,0],[200,0],[200,400]]

        self.transform_matrix = cv2.getPerspectiveTransform(np.float32(self.from_corners), np.float32(self.to_corners))

        print('Run the bash script on the Pi now')
        self.vid = cv2.VideoCapture('udp://0.0.0.0:10000?overrun_nonfatal=1&fifo_size=5000000')
        print('Connected to Pi')
        self.vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))
        self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = self.vid.read()[1]
        self.w = self.frame.shape[0]
        self.h = self.frame.shape[1]
        
        self.last_frame_time = time.time()  # seconds

        # Best parameters found
        self.camera_matrix = np.array([[283.08720786,   0.        , 319.49999987],
            [  0.        , 224.13115655, 239.49999971],
            [  0.        ,   0.        ,   1.        ]])

        self.distortion_coeffs = np.array([[-1.55053177e-02,  5.16288067e-05, -5.41872511e-03,
        -2.47583796e-03, -4.58942756e-08]])

        self.optimalcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, (self.w,self.h), 0, (self.w,self.h))

        self.found = 0
        self.tick_tocks = []

        self._create_blob_detector()


    def _create_blob_detector(self):
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 300
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7
        
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87
        
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01
        
        self.detector = cv2.SimpleBlobDetector_create(params)


    def publish_callback(self):
        print(self.puck_pos)


    def update_puck_status(self):
        frame = self.vid.read()[1]
        time_stamp = time.time()

        self.frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs, None, self.optimalcameramtx)
        # cv2.imshow('Distortion Corrected',self.frame)

        self.frame = cv2.warpPerspective(self.frame, M=self.transform_matrix, dsize=self.des_image_shape)

        keypoints = self.detector.detect(self.frame)
        # print(keypoints)

        if len(keypoints):
            x = keypoints[0].pt[0]
            y = keypoints[0].pt[1]
            im_with_keypoints = cv2.drawKeypoints(self.frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Keypoints", im_with_keypoints)

            if x > -1 and y > -1 and self.puck_pos[0] > -1 and self.puck_pos[1] > -1:
                vx = (x - self.puck_pos[0]) / (time_stamp - self.last_frame_time)
                vy = (y - self.puck_pos[1]) / (time_stamp - self.last_frame_time)

                epsilon = 0.6 #higher is mostly current [0; 1]
                self.puck_vel[0] = (1-epsilon)*self.puck_vel[0] + epsilon*vx
                self.puck_vel[1] = (1-epsilon)*self.puck_vel[1] + epsilon*vy
            else:
                self.puck_vel = [0, 0]

            self.puck_pos = keypoints[0].pt
        else:
            self.puck_pos = [-100, -100]
            self.puck_vel = [0, 0]
        
        cv2.imshow('Frame',self.frame)
        cv2.waitKey(1)

        self.last_frame_time = time_stamp
        print(self.puck_vel)


    def filter_for_puck_bgr(self):
        cv2.inRange(self.frame,self.lower_bgr_bound,self.upper_bgr_bound)


def main():
    puck_tracker = PuckTracker()

    try:
        while(True):
            puck_tracker.update_puck_status()
    except KeyboardInterrupt:
        puck_tracker.vid.release()
        
# f = open('data/data.csv', 'w', newline='')
# writer = csv.writer(f)

if __name__ == '__main__':
    main()
