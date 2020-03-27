"""
Created on Sun Sep 22 18:32:30 2019

@author: vince
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

im = Image.open("A1_Q4.jpg").convert('L')
imArray = np.asarray(im)
fullPhoto = Image.open("A1_Q4c.jpg").convert('L')
fullPhotoArray = np.asarray(fullPhoto)
face = Image.open("A1_Q4c_face.jpg").convert('L')
faceArray = np.asarray(face)
background = Image.open("A1_Q4c_background.jpg").convert('L')
backgroundArray = np.asarray(background)

test = [[1,2,1],
        [2,4,2],
        [1,2,1]]

test2 = [[0,0,1,0,0],
         [0,1,2,1,0],
         [1,2,4,2,1],
         [0,1,2,1,0],
         [0,0,1,0,0]]

test2 = [[0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,1,2,1,0,0,0,0,0,0,0],
         [1,2,4,2,1,0,0,0,0,0,0],
         [0,1,2,1,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0],
         [0,0,1,0,0,0,0,0,0,0,0]]

h = [[0,0,0],
     [0,0,1],
     [0,0,0]]

h1 = [[0,0,0],
      [0,1,0],
      [0,1,2],
      [0,1,0],
      [0,0,0]]

h2 = [[0,0,0,0,0],
      [0,0,0,0,1],
      [0,0,0,0,0]]

gauss = 1/256 * np.asarray([[1,4,6,4,1],[4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
sharpening = np.asarray([[0,0,0],[0,2,0],[0,0,0]]) - 1/9 * np.asarray([[1,1,1],[1,1,1],[1,1,1]])

def G(i, j, I, h):
    result = 0
    for u in range(-(len(h)//2), len(h)//2+1):
        for v in range(-(len(h[0])//2), len(h[0])//2+1):
            result += h[u+(len(h)//2)][v+(len(h[0])//2)] * I[i+u][j+v]
    return result

def MyCorrelation(I, h, mode):
    result = []
    if mode == 'valid':
        if len(h)<=len(I) or len(h[0])<=len(I[0]):
            for y in range(len(h)//2, len(I)-len(h)//2):
                temp = []
                for x in range(len(h[0])//2, len(I[0])-len(h[0])//2):
                    temp.append(G(y,x,I,h))
                result.append(temp)
    elif mode == 'same':
        newArray = np.zeros((len(I)+len(h)-1, len(I[0])+len(h[0])-1), int)
        for y in range(len(h)//2, len(newArray)-len(h)//2):
            temp = []
            for x in range(len(h[0])//2, len(newArray[0])-len(h[0])//2):
                newArray[y][x] = I[y-len(h)//2][x-len(h[0])//2]
             
        for y in range(len(h)//2, len(newArray)-len(h)//2):
            temp = []
            for x in range(len(h[0])//2, len(newArray[0])-len(h[0])//2):
                temp.append(G(y,x,newArray,h))
            result.append(temp)
    elif mode == 'full':
        newArray = np.zeros((len(I)+2*len(h)-2, len(I[0])+2*len(h[0])-2), int)
        for y in range(len(h)-1, len(newArray)-len(h)+1):
            temp = []
            for x in range(len(h[0])-1, len(newArray[0])-len(h[0])+1):
                newArray[y][x] = I[y-(len(h)-1)][x-(len(h[0])-1)]
             
        for y in range(len(h)//2, len(newArray)-len(h)//2):
            temp = []
            for x in range(len(h[0])//2, len(newArray[0])-len(h[0])//2):
                temp.append(G(y,x,newArray,h))
            result.append(temp)
    return np.asarray(result)

def MyConvolution(I, h, mode):
    newH = np.flip(np.flip(h,0),1)
    return MyCorrelation(I, newH, mode)

"""
print("MyCorrelation for valid\n", MyCorrelation(test2, h1, 'valid'))
print("what you should be aiming for valid\n", signal.convolve2d(test2, np.rot90(np.rot90(h1)), mode='valid'))
print("MyCorrelation for same\n", MyCorrelation(test2, h1, 'same'))
print("what you should be aiming for same\n", signal.convolve2d(test2, np.rot90(np.rot90(h1)), mode='same'))
print("MyCorrelation for full\n", MyCorrelation(test2, test2, 'full'))
print("what you should be aiming for full\n", signal.convolve2d(test2, np.rot90(np.rot90(test2)), mode='full'))
"""

def PortraitMode(f, b, size, depth):
    result = b
    gauss = 1/256 * np.asarray([[1,4,6,4,1],[4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1]])
    for i in range(depth):
        result = MyCorrelation(result, gauss, size)
    return f + result

im1 = Image.fromarray(PortraitMode(faceArray, backgroundArray, 'same', 5))
im1 = im1.convert("L")
im1.save("portrait.jpeg")

im2 = Image.fromarray(fullPhotoArray)
im2 = im2.convert("L")
im2.save("original.jpeg")

"""
im1 = Image.fromarray(MyCorrelation(imArray, gauss, 'valid'))
im1 = im1.convert("L")
im1.save("gauss.jpeg")
"""