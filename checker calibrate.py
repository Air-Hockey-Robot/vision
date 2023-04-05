import cv2
import numpy as np


stream_commad =  "udp://10.42.0.1:5000?overrun_nonfatal=1&fifo_size=50000000"
cap = cv2.VideoCapture(stream_commad)

def live_calibrate(camera, pattern_shape, n_matches_needed):
    """ Find calibration parameters as the user moves a checkerboard in front of the camera """
    print("Looking for %s checkerboard" % (pattern_shape,))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    example_3d = np.zeros((pattern_shape[0] * pattern_shape[1], 3), np.float32)
    example_3d[:, :2] = np.mgrid[0 : pattern_shape[1], 0 : pattern_shape[0]].T.reshape(-1, 2)
    points_3d = []
    points_2d = []
    while len(points_3d) < n_matches_needed:
        ret, frame = camera.read()
        assert ret
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(
            gray_frame, pattern_shape, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )
        cv2.imshow("camera", frame)
        if ret:
            points_3d.append(example_3d.copy())
            points_2d.append(corners)
            print("Found calibration %i of %i" % (len(points_3d), n_matches_needed))
            drawn_frame = cv2.drawChessboardCorners(frame, pattern_shape, corners, ret)
            cv2.imshow("calib", drawn_frame)
        cv2.waitKey(10)
    ret, camera_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
        points_3d, points_2d, gray_frame.shape[::-1], None, None
    )
    assert ret
    return camera_matrix, distortion_coefficients 

print(live_calibrate(cap,[50, 50], 10))