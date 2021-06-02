import cv2
import random
import numpy as np

""" def biggestContourI(contours, frame):
    maxVal = 0
    maxI = None
    for i in range(0, len(contours) - 1):
        cv2.drawContours(frame,contours, i, (0,255,0), 3)
        cv2.convexityDefects(contours, )
    return maxI """

cv2.namedWindow("preview")
cv2.namedWindow("result")
cv2.namedWindow("finish")
vc = cv2.VideoCapture(0)
width = vc.get(3)
height = vc.get(4)
if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
    frame2 = frame
else:
    rval = False
xa = random.randint(0, width)
ya = random.randint(0, height)

lower_green = np.array([75, 100, 50])
upper_green = np.array([94, 255, 255])
color_info = (0, 0, 255)
while rval:
    rval, frame = vc.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.circle(frame, (xa, ya), 5, (0, 0, 255), -1)
    # preparing the mask to overlay
    hsv = cv2.blur(hsv, (5, 5))
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(elements) > 0:
        c = max(elements, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 30:
            print(x, y, 'x, y')
            cv2.circle(frame, (int(x), int(y)), 5, color_info, 10)

    cv2.imshow("preview", frame)
    cv2.imshow("result", result)
    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyAllWindows()
vc.release()
