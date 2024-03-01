import numpy as np
import cv2
import glob
import pandas as pd

# Set the number of chessboard corners (inner corners)
nx = 10  # number of inside corners in x
ny = 7   # number of inside corners in y

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images from a folder
images = glob.glob('/home/ftdc/Documents/camera_calibration/logitic/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Convert imgpoints to numpy array
imgpoints_array = np.array(imgpoints, dtype=np.float32)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_array, gray.shape[::-1], None, None)

# Create a DataFrame to store camera calibration parameters
data = {
    "Camera matrix": [mtx],
    "Distortion coefficients": [dist],
    "Rotation vectors": [rvecs],
    "Translation vectors": [tvecs]
}
df = pd.DataFrame(data)

# Save DataFrame to Excel file
df.to_excel("camera_calibration_parameters.xlsx", index=False)
