"""
Extracted code from tutorial B
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread('building.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def Harris(image, alpha):
    blur = cv2.GaussianBlur(image,(5,5),7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    
    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10)
    
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    R = det - alpha * np.multiply(trace,trace)
    return R

def Brown(image, alpha):
    blur = cv2.GaussianBlur(image,(5,5),7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)
    
    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10)
    
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    R = det / (trace+alpha)
    return R

#Harris
R_H = Harris(gray, 0.04)
fig1, graph = plt.subplots(figsize=(10,5))
graph.imshow(R_H,cmap = 'gray')
graph.axis('off')
fig1.savefig('Q2aHarris.png')

#Brown
#smoother function in the region where lambda0 ~ lambda1
R_B = Brown(gray,0.04)
fig2, graph =plt.subplots(figsize=(10,5))
graph.imshow(R_B,cmap = 'gray')
graph.axis('off')
fig2.savefig('Q2aBrown.png')

#Part B
#https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
height, width = gray.shape[1::-1]
center = tuple(np.array(gray.shape[1::-1]) / 2)
filter1 = cv2.getRotationMatrix2D(center, 60, 1.0)
abs_cos = abs(filter1[0,0]) 
abs_sin = abs(filter1[0,1])
bound_w = int(height * abs_sin + width * abs_cos)
bound_h = int(height * abs_cos + width * abs_sin)
filter1[0, 2] += bound_w/2 - center[0]
filter1[1, 2] += bound_h/2 - center[1]
rotated = cv2.warpAffine(gray, filter1,(bound_w, bound_h))
R_H = Harris(rotated, 0.04)
fig3, graph = plt.subplots(figsize=(10,10))
graph.imshow(R_H,cmap = 'gray')
graph.axis('off')
fig3.savefig('Q2b.png')


#Part C
def LoG(image, sigma):
    result = ndimage.gaussian_laplace(image, sigma)

    return result
    
R_LoG = LoG(gray,7)
fig2, graph =plt.subplots(figsize=(10,5))
graph.imshow(R_LoG,cmap = 'gray')
graph.axis('off')
fig2.savefig('Q2c.png')

