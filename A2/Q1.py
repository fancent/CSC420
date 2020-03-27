import numpy as np
import cv2
from scipy import signal


#Question 1
beeImg = cv2.imread("bee.jpg")
b,g,r = cv2.split(beeImg)

def interpolation1D(image, size):
    """
    This takes an image and stretch it out along the a single axis
    """
    h = [i/size for i in range(size)] + [1] + [i/size for i in range(size-1,-1, -1)]
    result = []
    for i in range(len(image)):
        #padded = [0 for i in range((size-1)*(len(image[0])-1)+len(image[0]))]
        padded = [0. for i in range((len(image[0])-1)*size +1)]
        padded[::size] = image[i]
        padded = np.pad(padded, (size), 'constant')
        temp= []
        for j in range(size, len(padded)-size):
            sliced = padded[(j-size):(j+size+1)]
            product = np.matmul(h, sliced)
            temp.append(product)
        result.append(temp)
    return np.asarray(result)

bx4 = interpolation1D(b, 4)
gx4 = interpolation1D(g, 4)
rx4 = interpolation1D(r, 4)

by4 = interpolation1D(bx4.T, 4)
gy4 = interpolation1D(gx4.T, 4)
ry4 = interpolation1D(rx4.T, 4)

merged = cv2.merge((by4.T, gy4.T, ry4.T))
cv2.imwrite('Q1a.png', merged)

def interpolation2D(image, size):
    """
    This takes an image and stretch it out with a 2D kernal
    """
    h = [i/size for i in range(size)] + [1] + [i/size for i in range(size-1,-1, -1)]
    hT = [[i/size] for i in range(size)] + [[1]] + [[i/size] for i in range(size-1,-1, -1)]
    kernal = np.multiply(hT, h)
    
    padded = []
    for i in range(len(image)):
        temp = [0. for i in range((len(image[0])-1)*size +1)]
        temp[::size] = image[i]
        padded.append(temp)
        if i < len(image)-1:
            for j in range(size-1):
                temp2 = [0. for i in range((len(image[0])-1)*size +1)]
                padded.append(temp2)
    
    result = signal.convolve2d(padded, kernal, mode='same')
    return result


b = interpolation2D(b, 4)
g = interpolation2D(g, 4)
r = interpolation2D(r, 4)

merged = cv2.merge((b, g, r))
cv2.imwrite('Q1b.png', merged)
