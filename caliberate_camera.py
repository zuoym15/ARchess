import numpy as np
import cv2
import glob

from ARchess import frame_reader

import json

reader = frame_reader()

# Define the chess board rows and columns
rows = 7
cols = 7

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArray = []
imgPointsArray = []

valid_frame = 0


# Loop over the image files
#for path in glob.glob('../data/left[0-1][0-9].jpg'):
while True:
    # Load the image and convert it to gray scale
    #img = cv2.imread(path)
    ret, img = reader.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # Make sure the chess board pattern was found in the image
    if ret:
        # Refine the corner position
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Add the object points and the image points to the arrays
        objectPointsArray.append(objectPoints)
        imgPointsArray.append(corners)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, (rows, cols), corners, ret)

        piece = 'Valid Frame: ' + str(valid_frame+1) + '/30'

        #cv2.putText(img = img, text = piece, org = (gray.shape[0], gray.shape[1]), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 5, color = (255, 0, 0)) 

        print(piece) 

        valid_frame += 1
    
    # Display the image
    cv2.imshow('chess board', img)
    cv2.waitKey(1000)

    if valid_frame >= 30:
        break

# Calibrate the camera and save the results
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPointsArray, imgPointsArray, gray.shape[::-1], None, None)
print(mtx)
#np.savez('../data/calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Print the camera calibration error
error = 0

for i in range(len(objectPointsArray)):
    imgPoints, _ = cv2.projectPoints(objectPointsArray[i], rvecs[i], tvecs[i], mtx, dist)
    error += cv2.norm(imgPointsArray[i], imgPoints, cv2.NORM_L2) / len(imgPoints)

print("Total error: ", error / len(objectPointsArray))

with open('./camera_parameters.json', 'w') as f:
    json.dump(mtx.tolist(), f)
