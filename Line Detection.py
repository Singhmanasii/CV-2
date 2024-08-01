import cv2
import numpy as np

image = cv2.imread('C:/Users/singh/Downloads/lines.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(
    edges,
    rho=2,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=20,
    maxLineGap=5
)

line_list = []

if lines is not None:
    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        line_list.append([(x1, y1), (x2, y2)])

cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
