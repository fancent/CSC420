"""
Created on Sun Sep 22 23:48:30 2019

@author: vince
"""
import numpy as np

h1 = [[-1/8,0,1/8],
     [-2/8,0,2/8],
     [-1/8,0,1/8]]

h2 = [[1,0,-1],
     [2,0,-2],
     [1,0,-1]]

h3 = [[1/16,2/16,1/16],
     [2/16,4/16,2/16],
     [1/16,2/16,1/16]]

h4 = [[1/4,-2/4,1/4],
     [-2/4,4/4,-2/4],
     [1/4,-2/4,1/4]]

h5 = [[1/9,1/9,1/9],
     [1/9,1/9,1/9],
     [1/9,1/9,1/9]]

h6 = [[5,7,13],
     [23,11,17],
     [1,19,3]]


def isSeparableFilter(h):
    u, s, v = np.linalg.svd(h)
    nonZero = 0
    for i in s:
        if i >= 1e-15:
            nonZero += 1
    if nonZero == 1:
        print("horizontal 1D filter is:\n", np.dot(np.sqrt(s[0]), -v[0]))
        print("vertical 1D filter is:\n", np.dot(np.sqrt(s[0]), -u[:,0]))
        return True
    return False

print(isSeparableFilter(h5))
