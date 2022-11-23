import numpy as np
import cv2 as cv
import glob

chessboardSize = (9, 7)
frameSize = (1440, 1080)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0],
                       0:chessboardSize[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('/home/bhaswanth/Pictures/Webcam/webcam_*')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None)

print("Camera Calibrated", ret)
print("\nCamera Matrix:\n", cameraMatrix)
print("\nDistortion Parameters:\n", dist)
print("\nRotation Vectors:\n", rvecs)
print("\nTranslation Vectors:\n", tvecs)

img = cv.imread('/home/bhaswanth/Pictures/Webcam/webcam_test.jpg')

h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 1, (w, h))

dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

x, y, w, h = roi
cv.imwrite('caliResult0.jpg', dst)
dst = dst[y:y+h, x:x+w]
cv.imshow('img', img)
cv.imshow('Undistorted.jpg', dst)

dst = dst[y:y+h, x:x+w]
cv.imwrite('caliresult1.jpg', dst)

cv.waitKey(0)

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error/len(objpoints)))

cv.destroyAllWindows()
