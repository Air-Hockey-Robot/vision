import time
import numpy as np
import pickle
import os
from scipy.signal import savgol_filter 
import cv2


def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # draw circle here (etc...)
            print(f'[{x},{y}],')     


def get_corners():
    print('Run the script on the Pi now')
    vid = cv2.VideoCapture('udp://0.0.0.0:10000?overrun_nonfatal=1&fifo_size=5000000')
    print('connected')
    vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))

    frame = vid.read()[1]
    w = frame.shape[0]
    h = frame.shape[1]

    # Best parameters found
    camera_matrix = np.array([[283.08720786,   0.        , 319.49999987],
    [  0.        , 224.13115655, 239.49999971],
    [  0.        ,   0.        ,   1.        ]])
    distortion_coeffs = np.array([[-1.55053177e-02,  5.16288067e-05, -5.41872511e-03,
    -2.47583796e-03, -4.58942756e-08]])

    #     distortion_coeffs = np.array([[-1.43385286e-02,  4.59534890e-05,  7.82252201e-05,
    #      1.26979637e-04, -4.07249132e-08]])
        
    #     camera_matrix = np.array([[ 44.87445916,   0.        , 319.49999996],
    #    [  0.        , 177.86883669, 239.49999997],
    #    [  0.        ,   0.        ,   1.        ]])

    optimalcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w,h), 0, (w,h))

    # frame = frame
    frame = cv2.undistort(frame, camera_matrix, distortion_coeffs, None, optimalcameramtx)
    cv2.imshow('undistorted',frame)
    
    cv2.setMouseCallback('undistorted', onMouse)

    cv2.waitKey()


def main(args=None):
    print('[')
    get_corners()
    print(']')


if __name__ == '__main__':
    main()
