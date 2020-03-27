"""
Created on Mon Sep 23 23:00:08 2019

@author: vince
"""

import numpy as np
from PIL import Image
import cv2

img = Image.open("gray.jpg")
imgArray = 1./255 * np.asarray(img, dtype="float32")

def AddRandNoise(I, m):
    noise = np.random.uniform(-m,m,(I.shape[0],I.shape[1])).astype("float32")
    result = I + noise
    return result/(1/255)

noiseImage = AddRandNoise(imgArray,0.05)

cv2.imwrite('A1_Q6_a.jpg', noiseImage)

filtered = cv2.GaussianBlur(noiseImage,(3,3),0)

cv2.imwrite('A1_Q6_b.jpg', filtered)

filtered = cv2.medianBlur(noiseImage,3)

cv2.imwrite('A1_Q6_b1.jpg', filtered)

def AddSaltAndPepperNoise(I, d):
    for i in range(int(d*len(I)*len(I[0]))):
        y = np.random.randint(0, len(I)-1)
        x = np.random.randint(0, len(I[0])-1)
        I[y][x] = np.random.choice([0,1])
    return I

saltPepperImage = AddSaltAndPepperNoise(imgArray/(1/255), 0.05)
cv2.imwrite('A1_Q6_c.jpg', saltPepperImage)

filteredSaltAndPepper = cv2.medianBlur(noiseImage,1)

cv2.imwrite('A1_Q6_d.jpg', filteredSaltAndPepper)

colorImg = cv2.imread("color.jpg")
b,g,r = cv2.split(colorImg)

bSaltPepper = AddSaltAndPepperNoise(b, 0.05)
gSaltPepper = AddSaltAndPepperNoise(g, 0.05)
rSaltPepper = AddSaltAndPepperNoise(r, 0.05)

merged = cv2.merge((bSaltPepper, gSaltPepper, rSaltPepper))
cv2.imwrite('A1_Q6_e1.jpg', merged)

bfiltered = cv2.medianBlur(bSaltPepper,3)
gfiltered = cv2.medianBlur(gSaltPepper,3)
rfiltered = cv2.medianBlur(rSaltPepper,3)

mergedFiltered = cv2.merge((bfiltered, gfiltered, rfiltered))
cv2.imwrite('A1_Q6_e2.jpg', mergedFiltered)


h = 1/9 * np.asarray([[1,1,1],[1,1,1],[1,1,1]])
def G(i, j, I, h):
    result = 0
    for u in range(-(len(h)//2), len(h)//2+1):
        for v in range(-(len(h[0])//2), len(h[0])//2+1):
            result += h[u+(len(h)//2)][v+(len(h[0])//2)] * I[i+u][j+v]**2
    return np.sqrt(result)

def MyCorrelation(I, h):
    result = []
    newArray = np.zeros((len(I)+len(h)-1, len(I[0])+len(h[0])-1,3), int)
    for y in range(len(h)//2, len(newArray)-len(h)//2):
        temp = []
        for x in range(len(h[0])//2, len(newArray[0])-len(h[0])//2):
            newArray[y][x] = I[y-len(h)//2][x-len(h[0])//2]
         
    for y in range(len(h)//2, len(newArray)-len(h)//2):
        temp = []
        for x in range(len(h[0])//2, len(newArray[0])-len(h[0])//2):
            temp.append(G(y,x,newArray,h))
        result.append(temp)
    return np.asarray(result)

cv2.imwrite('A1_Q6_e3.jpg', MyCorrelation(mergedFiltered,h))
