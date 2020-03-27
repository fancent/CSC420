import numpy as np
import cv2
import matplotlib.pyplot as plt

def getScoreImg():
    nums = cv2.imread("numbers.png")
    ret, nums = cv2.threshold(nums,127,255,cv2.THRESH_BINARY_INV)
    nums = cv2.cvtColor(nums, cv2.COLOR_BGR2GRAY)
    top = np.zeros((1,530))
    nums = np.concatenate((top,top,top,top, nums, top,top,top,top,top), axis=0)
    nums = cv2.resize(nums, (530*40//109, 40))
    ret, nums = cv2.threshold(nums,127,255,cv2.THRESH_BINARY_INV)

    zero = nums[:, 2:20]
    one = nums[:, 21:38]
    two = nums[:, 39: 57]
    three = nums[:, 58: 77]
    four = nums[:, 76:97]
    five = nums[:, 96:115]
    six = nums[:, 115:134]
    seven = nums[:, 134:153]
    eight = nums[:, 152:172]
    nine = nums[:, 171:191]
    cv2.imwrite("0.png", zero)
    cv2.imwrite("1.png", one)
    cv2.imwrite("2.png", two)
    cv2.imwrite("3.png", three)
    cv2.imwrite("4.png", four)
    cv2.imwrite("5.png", five)
    cv2.imwrite("6.png", six)
    cv2.imwrite("7.png", seven)
    cv2.imwrite("8.png", eight)
    cv2.imwrite("9.png", nine)

    plt.imshow(one, cmap='gray')
    plt.show()

def getTimeImg():
    nums = cv2.imread("time.png")
    ret, nums = cv2.threshold(nums,127,255,cv2.THRESH_BINARY_INV)
    nums = cv2.cvtColor(nums, cv2.COLOR_BGR2GRAY)
    print(nums.shape)
    # top = np.zeros((1,530))
    # nums = np.concatenate((top,top,top,top,top,top,top,top, nums,top,top,top,top,top,top,top,top,top), axis=0)
    nums = cv2.resize(nums, (480*40//172, 40))
    ret, nums = cv2.threshold(nums,127,255,cv2.THRESH_BINARY_INV)

    zero = nums[:, 0:12]
    one = nums[:, 12:21]
    two = nums[:, 22: 33]
    three = nums[:, 33: 44]
    four = nums[:, 43:56]
    five = nums[:, 55:66]
    six = nums[:, 66:78]
    seven = nums[:, 77:88]
    eight = nums[:, 88:100]
    nine = nums[:,99:110]
    cv2.imwrite("time_0.png", zero)
    cv2.imwrite("time_1.png", one)
    cv2.imwrite("time_2.png", two)
    cv2.imwrite("time_3.png", three)
    cv2.imwrite("time_4.png", four)
    cv2.imwrite("time_5.png", five)
    cv2.imwrite("time_6.png", six)
    cv2.imwrite("time_7.png", seven)
    cv2.imwrite("time_8.png", eight)
    cv2.imwrite("time_9.png", nine)

    plt.imshow(nine, cmap='gray')
    plt.show()

if __name__ == "__main__":
    getScoreImg()
    # getTimeImg()