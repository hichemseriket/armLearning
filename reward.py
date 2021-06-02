import cv2
import numpy as np
import datetime
import random


def souris(event, x, y, flags, param):
    global lo, hi, color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        color = image[y, x][0]
        colors.append(image[y, x])
    if event == 1:
        setHighAndLow(image[y, x])
    elif event == 2:
        lo = lastLo
        hi = lastHi


def sendToCode(value):
    print('TODO send ', value)


def setHighAndLow(element):
    global minH, minS, minV, maxH, maxS, maxV, lo, hi, lastHi, lastLo
    lastHi = hi
    lastLo = lo
    if (element[0] < minH):
        minH = element[0]
    if (element[0] > maxH):
        maxH = element[0]
    if (element[1] < minS):
        minS = element[1]
    if (element[1] > maxS):
        maxS = element[1]
    if (element[2] < minV):
        minV = element[2]
    if (element[2] > maxV):
        maxV = element[2]
    lo = np.array([minH - 5, minS, minV])
    hi = np.array([maxH + 5, maxS, maxV])


# Permet de calibrer la couleur à suivre
minH = 255
minS = 255
minV = 255
maxH = 0
maxS = 0
maxV = 0
color = 100
lo = np.array([color - 5, 100, 50])
hi = np.array([color + 5, 255, 255])
lastLo = np.array([0, 0, 0])
lastHi = np.array([0, 0, 0])
color_info = (0, 0, 255)
cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', souris)
print('config low : ', lo, ' high: ', hi)
colors = []
while True:
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    image = cv2.blur(image, (5, 5))
    mask = cv2.inRange(image, lo, hi)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    image2 = cv2.bitwise_and(frame, frame, mask=mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# permet de définir les valeurs min et max
print('fin de la définition de la couleur')

# run du jeu
# definition du point d'arrivée
width = cap.get(3)
height = cap.get(4)
date = datetime.datetime.now()
tour = 0


def runGame(tour):
    if (tour < 3):
        xa = random.randint(0, width)
        ya = random.randint(0, height)
        print(lo, hi, 'high low')
        nbr_point = 100
        tab_point = np.full((nbr_point, 2), -1, dtype=np.int32)
        lastDistance = 100000000
        while True:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            image = cv2.blur(image, (5, 5))
            mask = cv2.inRange(image, lo, hi)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=1)
            frame = cv2.circle(frame, (xa, ya), radius=5, color=(0, 0, 255), thickness=-1)
            image2 = cv2.bitwise_and(frame, frame, mask=mask)
            elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            tab_point = np.roll(tab_point, 1, axis=0)
            tab_point[0] = [-1, -1]
            if len(elements) > 0:
                c = max(elements, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if radius > 30:
                    print(x, y, 'x, y')
                    newDistance = ((((x - xa) ** 2) + ((y - ya) ** 2)) ** 0.5)
                    if int(newDistance) >= 10:
                        tab_point[0] = [int(x), int(y)]
                        if (lastDistance > newDistance):
                            sendToCode('+')
                        else:
                            sendToCode('-')
                    else:
                        sendToCode('=')
                        break
                    lastDistance = newDistance
                    cv2.circle(frame, (int(x), int(y)), 5, color_info, 10)
                    lastDistance, newDistance = (-1, -1)
                    for i in range(nbr_point):
                        if tab_point[nbr_point - i - 1, 0] != -1:
                            if lastDistance != -1:
                                cv2.line(frame, (lastDistance, newDistance),
                                         (tab_point[nbr_point - i - 1, 0], tab_point[nbr_point - i - 1, 1]),
                                         (0, 255 - 2 * (nbr_point - i - 1), 0), 10)
                                # cv2.line(frame, (old_x, old_y), (tab_point[nbr_point-i-1, 0], tab_point[nbr_point-i-1, 1]), (0, 255, 0), 10)
                            lastDistance, newDistance = (tab_point[nbr_point - i - 1, 0], tab_point[nbr_point - i - 1, 1])

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        runGame(tour + 1)


runGame(tour)
cap.release()
cv2.destroyAllWindows()
