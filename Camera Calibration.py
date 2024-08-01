import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('C:/Users/singh/Downloads/chess*.jpeg')

if not images:
    print("No images found. Check the path and file pattern.")
else:
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(f"Chessboard corners not found in image {fname}")

    cv.destroyAllWindows()

    print(f'Number of valid object points: {len(objpoints)}')
    print(f'Number of valid image points: {len(imgpoints)}')

    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
        print("Rotation vectors:")
        print(rvecs)
        print("Translation vectors:")
        print(tvecs)

        img = cv.imread("C:/Users/singh/Downloads/chess camera calib.jpeg")
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite('calibresult.jpg', dst)
        cv.imshow('calibresult', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Not enough points for calibration. Check your images and detection process.")
