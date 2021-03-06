import cv2
import numpy as np
import datetime
import random
from random import randint

color_infos = (0, 0, 255)


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

# couleur pour detecter l'arbre a ma fenetre
# lo = np.array([color - 5, 100, 50])
lo = np.array([color - 80, 15, 75])
# hi = np.array([color + 5, 255, 255])
hi = np.array([color + 15, 50, 195])
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
        mode = 1
        degrade = 1

        def dessine_point(tab_point):
            for i in range(len(tab_point)):
                if tab_point[nbr_point - i - 1, 0] != -1:
                    if degrade:
                        couleur = (0, 255 - 2 * (nbr_point - i - 1), 0)
                    else:
                        couleur = (0, 255, 0)
                    cv2.circle(frame, (tab_point[nbr_point - i - 1, 0], tab_point[nbr_point - i - 1, 1]), 5, couleur,
                               10)

        def dessine_ligne(tab_point):
            lastDistance, newDistance = (-1, -1)
            for i in range(nbr_point):
                if tab_point[nbr_point - i - 1, 0] != -1:
                    if lastDistance != -1:
                        if degrade:
                            couleur = (0, 255 - 2 * (nbr_point - i - 1), 0)
                        else:
                            couleur = (0, 255, 0)
                        cv2.line(frame, (lastDistance, newDistance),
                                 (tab_point[nbr_point - i - 1, 0], tab_point[nbr_point - i - 1, 1]),
                                 couleur, 10)
                lastDistance, newDistance = (tab_point[nbr_point - i - 1, 0], tab_point[nbr_point - i - 1, 1])

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
                    cv2.line(frame, (int(x), int(y)), (int(x) + 150, int(y)), color_infos, 2)

                    cv2.putText(frame, "Objet !!!", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color_infos,
                                1, cv2.LINE_AA)

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
                    if mode:
                        dessine_ligne(tab_point)
                    else:
                        dessine_point(tab_point)
                    cv2.rectangle(frame, (0, 0), (int(width), 30), (100, 100, 100), cv2.FILLED)
                    # cv2.putText(mask, "x: {:d}  y: {:d}".format(lo, hi))
                    cv2.putText(frame, "Mode[m]: {:d}   Degrade[p]: {:d}".format(mode, degrade), (10, 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            cv2.imshow('Camera', frame)
            cv2.imshow('HICHEM', image2)
            cv2.imshow('Mask', mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                mode = not mode
            if key == ord('p'):
                degrade = not degrade
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        runGame(tour + 1)


runGame(tour)
class EnvGrid(object):

    def __init__(self):
        super(EnvGrid, self).__init__()

# l'environnement n'est pas une grille mais bon on peut dire que notre camera est une grille,
# et que a chaque mouvement l'environnement change et devient une nouvelle grille
        self.grid = [
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0]
        ]
        # Starting position
        self.y = 2
        self.x = 0

# les actions je vais quand meme mettre uniquement les deux : s'approcher, s'eloigner
        self.actions = [
            [-1, 0], # Up
            [1, 0], #Down
            [0, -1], # Left
            [0, 1] # Right
        ]

    def reset(self):
        """
            Reset world
        """
        self.y = 2
        self.x = 0
        return (self.y*3+self.x+1)

    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        self.y = max(0, min(self.y + self.actions[action][0],2))
        self.x = max(0, min(self.x + self.actions[action][1],2))

        return (self.y*3+self.x+1) , self.grid[self.y][self.x]

    def show(self):
        """
            Show the grid
        """
        print("---------------------")
        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                print("%s\t" % (pt if y != self.y or x != self.x else "X"), end="")
                x += 1
            y += 1
            print("")

    def is_finished(self):
        return self.grid[self.y][self.x] == 1

def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        action = randint(0, 3)
    else: # Or greedy action
        action = np.argmax(Q[st])
    return action

if __name__ == '__main__':
    env = EnvGrid()
    st = env.reset()

    Q = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    for _ in range(100):
        # Reset the game
        st = env.reset()
        while not env.is_finished():
            #env.show()
            #at = int(input("$>"))
            at = take_action(st, Q, 0.4)

            stp1, r = env.step(at)
            #print("s", stp1)
            #print("r", r)

            # Update Q function
            atp1 = take_action(stp1, Q, 0.0)
            Q[st][at] = Q[st][at] + 0.1*(r + 0.9*Q[stp1][atp1] - Q[st][at])

            st = stp1

    for s in range(1, 10):
        print(s, Q[s])

cap.release()
cv2.destroyAllWindows()
