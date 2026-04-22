import numpy as np 
import cv2 
import glob 
# Define the number of corners in the chessboard 
num_corners_x = 9 
num_corners_y = 6 
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) 
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32) 
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2) 
# Arrays to store object points and image points from all the images 
objpoints = []  # 3d point in real world space 
imgpoints = []  # 2d points in image plane. 
# Load images 
images = glob.glob(r"C:\cv practicals\Calibration_images\*.jpeg")
 
# Loop through images and find chessboard corners 
for fname in images: 
    img = cv2.imread(fname) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Find the chessboard corners 
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None) 
 
    # If found, add object points, image points (after refining them) 
    if ret: 
        objpoints.append(objp) 
 
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)) 
        imgpoints.append(corners2) 
 
        # Draw and display the corners 
        img = cv2.drawChessboardCorners(img, (num_corners_x, num_corners_y), corners2, ret) 
        cv2.imshow('img', img) 
        cv2.waitKey(0) 
 
cv2.destroyAllWindows() 
 
# Perform camera calibration 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) 
 
# Save calibration results 
np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs) 
 
# Print calibration results 
print("Camera matrix:") 
print(mtx) 
print("\nDistortion coefficients:") 
print(dist)
