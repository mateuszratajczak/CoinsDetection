import numpy as np
import math  
import cv2
import matplotlib.pyplot as plt
import sys

class CoinTemp:
    def __init__(self, value, imagePath):
        self.value = value
        self.image = cv2.imread(imagePath)

class Coin:
    def __init__(self, circle):
        self.circle = circle
        self.count = 0
        self.values = []

    def inc(self):
        self.count += 1

    def getX(self):
        return self.circle[0]

    def getY(self):
        return self.circle[1]

    def getRadius(self):
        return self.circle[2]

    def checkPoint(self, point):
        dist = math.sqrt((point[0] - self.getX())**2 + (point[1] - self.getY())**2)
        return dist < self.getRadius()
    
    def apply(self, value):
        self.values.append((value, self.count))
        self.count = 0
    
    def getValue(self):
        result = 0
        maxC = 0
        for v in self.values:
            if(v[1] > maxC):
                maxC = v[1]
                result = v[0]
        return result

def getCircles(img):
    w = len(img0)
    h = len(img0[0])
    m = math.sqrt(w*h)
    print("m=" + str(m))
    p1 = 200 #200
    dp = 1000.0/m
    p2 = 170 #170
    minDist = m*0.25
    img = cv2.resize(img, (int(h*dp), int(w*dp)))
    img = cv2.medianBlur(img,31)
    #plt.imshow(img)
    #plt.show()
    count = 0
    while(count < 1):
        print("p1=" + str(p1))
        print("p2=" + str(p2))
        result = cv2.HoughCircles(img0,cv2.HOUGH_GRADIENT,1,minDist,
                            param1=p1,param2=p2,minRadius=0,maxRadius=0)
        p2 -= 20
        print(result)
        try:
            count = len(result[0])
        except:
            count = 0
    return result

try:
    aa2 = sys.argv[1]
    print(aa2)
except:
    aa2 = "zdjecie1.jpg"


templates = [
    CoinTemp(1, '1zl_bez_tla.png'),
    CoinTemp(1, '1zl_ver2.png'),
    CoinTemp(2, '2zl_bez_tla.png'),
    CoinTemp(2, '2zl_ver2.png'),
    CoinTemp(5, '5zl_bez_tla.png'),
    CoinTemp(5, '5zl_ver2.png')
]

imagePath = aa2;

img0 = cv2.imread(imagePath, 0)          # queryImage
img = cv2.imread(imagePath) 
#img0 = cv2.medianBlur(img0,5)
cimg = cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)

circles = getCircles(img0)
circles = np.uint16(np.around(circles))
coins = [Coin(circle) for circle in circles[0]]
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
imgPoints, imgDescriptors = sift.detectAndCompute(img,None)

points = []
for temp in templates:
    tempPoints, tempDescriptors = sift.detectAndCompute(temp.image,None)
    matches = flann.knnMatch(tempDescriptors,imgDescriptors,k=2)
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            point = imgPoints[m.trainIdx].pt
            for coin in coins:
                if(coin.checkPoint(point)):
                    points.append(point)
                    coin.inc()
    
    for coin in coins:
        coin.apply(temp.value)

sum = 0
for coin in coins:
    center = (coin.getX(), coin.getY())
    cv2.circle(img, center, coin.getRadius(), (0, 255, 0), 4)
    cv2.putText(img, str(coin.getValue()), center, 1, coin.getRadius()*0.075, (0, 255, 0), thickness=3)
    sum += coin.getValue()

for p in points:
    center = tuple(np.uint16(np.around((p[0], p[1]))))
    cv2.circle(img, center, 1, (255, 0, 0), 4)

plt.title("Wykryto łącznie " + str(sum) + " zł")
plt.imshow(img)
plt.show()
