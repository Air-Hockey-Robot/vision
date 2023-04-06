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
        self.frame_count = 0
        self.dir_path = os.path.dirname(os.path.realpath(__file__))  # directory of this python file

        self.serverAddressPort = ("192.168.0.102", 20001)
        self.bufferSize = 1024
        self.UDPSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

        with open(self.dir_path + '/camera_calib.pkl', 'rb') as f:
            self.from_corners = pickle.load(f)
        
        self.index = 1
        self.puck_pos = [-1.0, -1.0]
        self.puck_vel = [0.0, 0.0]

        self.show_frame = False
        self.SG_window = 50
        self.SV_poly_order = 4
        self.xvel_buffer = [0]*self.SG_window
        self.yvel_buffer = [0]*self.SG_window

        self.puck_radius =  3
        # print("M00 cut")
        self.y_dist = 640#40+13.0/16
        self.x_dist = 480#30+13.0/16
        self.max_size = 400
        
        self.M00_cut = 0.5*np.pi*(self.puck_radius*self.max_size/self.y_dist)**2
        # print(self.M00_cut)
        
        self.des_image_shape = [200, 400]
        self.pixels_to_cm = 0.5
        # self.from_corners = [[639,425],[2,436],[3,98],[639,126]]
        self.from_corners = [[638+2,426+2],[1,433],[4-3,94-4],[638+2,122+2]]
        self.to_corners = [[0,400],[0,0],[200,0],[200,400]]

        # print(self.to_corners)
        # print(self.from_corners)

        self.transform_matrix = cv2.getPerspectiveTransform(np.float32(self.from_corners), np.float32(self.to_corners))

        # self.vid = cv2.VideoCapture('udp://@192.168.0.102:5000?overrun_nonfatal=1&fifo_size=50000000')
        print('Run the script on the Pi now')
        self.vid = cv2.VideoCapture('udp://0.0.0.0:10000?overrun_nonfatal=1&fifo_size=5000000')
        print('connected')
        self.vid.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('H','2','6','4'))
        self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = self.vid.read()[1]
        self.w = self.frame.shape[0]
        self.h = self.frame.shape[1]
        
        self.last_frame_time = time.time()  # seconds


    # Best parameters found
        self.camera_matrix = np.array([[4.35468287e+02, 0.00000000e+00, 3.19500000e+02],
       [0.00000000e+00, 1.43591345e+03, 2.39500000e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    #     self.distortion_coeffs = np.array([[-1.43385286e-02,  4.59534890e-05,  7.82252201e-05,
    #      1.26979637e-04, -4.07249132e-08]])
        
    #     self.camera_matrix = np.array([[ 44.87445916,   0.        , 319.49999996],
    #    [  0.        , 177.86883669, 239.49999997],
    #    [  0.        ,   0.        ,   1.        ]])

        self.distortion_coeffs = np.array([[-7.31695304e-03,  1.34168701e-05, -3.90558151e-04,
        -4.98722075e-04, -6.84746782e-09]])

        self.optimalcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coeffs, (self.w,self.h), 0, (self.w,self.h))

        self.found = 0
        self.tick_tocks = []

        self.upper_bgr_bound = (108,92,164)
        self.lower_bgr_bound = (46,46,130)

        # cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('frame', 200   , 100)
        # time.sleep(2)


    def publish_callback(self):
        print(self.puck_pos)


    def update_puck_status(self):
        tic1 = time.time()
        tic = time.time_ns()
        ret, frame = self.vid.read()
        time_stamp = time.time()

        tic_pers = time.time()

        # self.frame = frame
        self.frame = cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs, None, self.optimalcameramtx)
        cv2.imshow('undistorted',self.frame)
        # cv2.waitKey()
        self.frame = cv2.warpPerspective(self.frame, M=self.transform_matrix, dsize=self.des_image_shape)
        # self.frame_green = self.frame[:,:,2] # TODO: change name to red later
        tock_pers = time.time()
        t_pers = (tock_pers-tic_pers)*1000
        # print(f"perspective transform took {t_pers} ms")
        # time_stamp = time.time()

        tic_filt = time.time()
        bin_img = self.filter_for_puck()
        # bin_img = self.filter_g()

        tock_filt = time.time()
        t_filt = (tock_filt-tic_filt)*1000
        # print(f"filtering took {t_filt} ms")
        self.bin_image = bin_img
        # self.display()
        # print(self.bin_image)

        tic_mom = time.time()
        M = cv2.moments(bin_img)
        # print(M['m00'])

        if M["m00"] < self.M00_cut:
            M["m00"] = 0
            self.found = self.found+1
            if self.found > 2:
            # if puck is lost for more than 3 frames then publish lost puck
                cX = -1.0
                cY = -1.0
            else: # otherwise chill pretend its still where it was
                cX = self.puck_pos[0]
                cY = self.puck_pos[1]
        else:
            cX = float(M["m10"] / M["m00"]) * self.pixels_to_cm
            cY = (self.des_image_shape[1] - float(M["m01"] / M["m00"])) * self.pixels_to_cm  # Subtracting from image height to get y=0 at bottom
            self.found = 0

        self.bin_image = cv2.cvtColor(self.bin_image, cv2.COLOR_GRAY2BGR)
        cv2.circle(self.bin_image, (int(cX/self.pixels_to_cm), int(self.des_image_shape[1] - cY/self.pixels_to_cm)), 5, (255, 0, 0), -1)

        tock_mom = time.time()
        t_mom = (tock_mom-tic_mom)*1000
        # print(f"moment calculations took {t_mom} ms")

        cv2.imshow("binary", self.bin_image)
        # cv2.waitKey(1)

        # print(self.frame.shape)
        cv2.imshow('frame',self.frame)
        # def onMouse(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
                # draw circle here (etc...)
        #         print('x = %d, y = %d'%(x, y))
        # cv2.setMouseCallback('frame', onMouse)
        # if self.frame_count % 100:
        
        cv2.waitKey(1)

        # if (self.puck_pos[0] is not None):
        #     # load new x and y velocities into buffer, and apply savgol filter to smooth noise
        #     del self.xvel_buffer[0]
        #     self.xvel_buffer.append((cX - self.puck_pos[0])/(time_stamp - self.last_frame_time))
        #     del self.yvel_buffer[0]
        #     self.yvel_buffer.append((cY - self.puck_pos[1])/(time_stamp - self.last_frame_time))
        #     xvel_filtered = savgol_filter(self.xvel_buffer, self.SG_window, self.SV_poly_order)
        #     yvel_filtered = savgol_filter(self.yvel_buffer, self.SG_window, self.SV_poly_order)
        #     self.puck_vel = [xvel_filtered[-1], yvel_filtered[-1]]
        vx = (cX - self.puck_pos[0])/(time_stamp - self.last_frame_time)
        vy = (cY - self.puck_pos[1])/(time_stamp - self.last_frame_time)
        epsilon = 0.6 #higher is mostly current [0; 1]
        self.puck_vel[0] = (1-epsilon)*self.puck_vel[0] + epsilon*vx
        self.puck_vel[1] = (1-epsilon)*self.puck_vel[1] + epsilon*vy

        self.puck_pos = [cX, cY]
        self.last_frame_time = time_stamp
        # self.publish_callback()
        toc = time.time_ns()
        self.tick_tocks.append(toc-tic)

        self.frame_count += 1
        if (self.frame_count % 10 == 0):
            print('pos:', self.puck_pos)
            # print('vel:', self.puck_vel[0])
        
        writer.writerow([time.time() - self.start_time, self.puck_pos[0], self.puck_pos[1]])

        # msg = str.encode("{:.3f} {:.3f} {:.3f} {:.3f}".format(*self.puck_pos, *self.puck_vel))
        # self.UDPSocket.sendto(msg, self.serverAddressPort)
        
        tock1 = time.time()
        # print((tock1-tic1)*1000)


    def filter_for_puck(self):
        # Convert to HSV and filter to binary image for puck isolation
        # Red puck has hue on boundary between 0 and 180, so two filters are used and summed
        hsv_img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsv_min = (0, 0, 10)
        hsv_max = (7, 255, 70)
        bin_img = (cv2.inRange(hsv_img, hsv_min, hsv_max))
        # hsv_min = (170, 80, 130)
        # hsv_max = (181, 143, 165)
        # high_hue_bin_img = (cv2.inRange(hsv_img, hsv_min, hsv_max))
        # bin_img = (low_hue_bin_img) | (high_hue_bin_img)
        return bin_img
        # # hsv_min = (0, 90, 45)
        # # hsv_max = (180, 140, 80)
        # return cv2.inRange(hsv_img, hsv_min, hsv_max)

    def filter_for_puck_bgr(self):
        cv2.inRange(self.frame,self.lower_bgr_bound,self.upper_bgr_bound)


def main(args=None):
    puck_tracker = PuckTracker()

    try:
        while(True):
            puck_tracker.update_puck_status()
    except KeyboardInterrupt:
        # close video write
        puck_tracker.vid.release()
        tictocs = np.array(puck_tracker.tick_tocks)
        plt.hist(tictocs/1e6)
        # print("Mean: {} ms".format(np.mean(tictocs/1e6)))
        plt.show()
        
f = open('data/data.csv', 'w', newline='')
writer = csv.writer(f)

if __name__ == '__main__':
    main()
