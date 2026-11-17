# Camera clibration using checkerboard method. Instruct users to do a one-time calibration with the same checkerboard pattern
# used for each user.

# 5/21/2026: Base implementation from OpenCV documentation.
# Add in future to save user/camera-specific calibration data and add error handling for failed corner detection.

# 5/24/2026: Note with implementation: We can have either calibrate in-app (see the component function) and
# have that component return the calibration parameters, or we send back each taken image to this script for processing

import cv2
import numpy as np
import os # file handling, may not be necessary off rip
import glob

term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHECKERBOARD = (6, 7) # Change dimensions when we settle on a pattern. Do NOT include points along the edge of the pattern.

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpts = [] # 3d physical points
imgpts = [] # 2d image points

images = glob.glob('*.jpg') # Change path as needed

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners. Note the tuple order is (col, row) per their documentation.
    ret, corners = cv2.findChessboardCorners(gray, (CHECKERBOARD[1], CHECKERBOARD[0]), None)

    if ret == True:
        objpts.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
        imgpts.append(corners2)

        # Draw and display the corners (debug)
        cv2.drawChessboardCorners(img, (CHECKERBOARD[1], CHECKERBOARD[0]), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, gray.shape[::-1], None, None)
img = cv2.imread('left12.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort (debug)
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

# Projetion error (quality assessment)
mean_error = 0.0
for i in range(len(objpts)):
    imgpts2, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpts[i], imgpts2, cv2.NORM_L2)/len(imgpts2)
    mean_error += error

print("Total error: {}".format(np.sqrt(mean_error/len(objpts))))