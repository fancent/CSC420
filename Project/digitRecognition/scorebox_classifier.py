import numpy as np
import cv2
import matplotlib.pyplot as plt

def getScoreBox(image):
    img = image.copy()
    img = img[581:621,240:1030]
    # img = img[605:645, 240:1030]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    ret, masked = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    plt.imshow(masked, cmap='gray')
    plt.show()
    cv2.imwrite("scorebox_6.png", masked)

def computeScore(template, image_patch):
    if template.shape != image_patch.shape:
        print(template.shape, image_patch.shape)
        raise ValueError("size does not match")
    score = np.sum((template - image_patch)**2)
    return score

def fastTemplateMatching(template, image, plot=False):
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    w,h = gray_template.shape[::-1]
    left, right = 240,1030
    img = gray_image[:, left:right]
    img = np.float32(img)
    ret, masked = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

    start = masked.shape[0] - masked.shape[0]//3
    end = masked.shape[0] - h
    score_list = []
    for i in range(start, end):
        score = computeScore(gray_template, masked[i:i+h,:])
        score_list.append(score)

    location = start + np.argmin(np.array(score_list))
    image[location:location+h, left:right] = [255,0,0]
    if plot:
        plt.plot(score_list)
        plt.show()
        plt.imshow(image)
        plt.show()
    return location, location+h

def randomTemplateMatching(templates, image):
    choices = np.random.choice(len(templates), 2)
    location1_start, location1_end = fastTemplateMatching(templates[choices[0]], image)
    location2_start, location2_end = fastTemplateMatching(templates[choices[1]], image)
    return int((location1_start+location2_start)/2), int((location1_end + location2_end)/2)

if __name__ == "__main__":
    temp1 = cv2.imread("scorebox_1.png")
    temp2 = cv2.imread("scorebox_2.png")
    temp3 = cv2.imread("scorebox_3.png")
    pic = cv2.imread("frame005.png")
    print(fastTemplateMatching(temp3, pic))
    print(randomTemplateMatching([temp1, temp2, temp3], pic))

# blurred = cv2.medianBlur(masked,5)
# blurred = cv2.medianBlur(blurred,5)
# blurred = cv2.medianBlur(blurred,5)
# blurred = cv2.medianBlur(blurred,5)


# laplacian = cv2.Laplacian(gray,cv2.CV_64F)
# laplacian = cv2.Laplacian(laplacian,cv2.CV_64F)

# edges = cv2.Canny(masked,100,200)

# dst = cv2.cornerHarris(edges,20,3,0.04)
# dst = cv2.dilate(dst,None)
# img[dst>0.01*dst.max()]=[0,0,255]

# print(edges)

