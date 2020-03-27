import numpy as np
import cv2
import matplotlib.pyplot as plt

def SSD(image, patch,start,end):
    results = []
    for i in range(start, end):
        sumDiff = np.sum((image[:,i:patch.shape[1]+i] - patch)**2)/(patch.shape[0]*patch.shape[1])
        results.append(sumDiff)
    results = np.array(results)
    score = np.min(results)
    first_index = start + np.argmin(results)
    second_index = first_index + patch.shape[1]-1
    return first_index, second_index, score

def getScores(image, num_list, start, end, showGraph=False):
    results = []
    score = 0
    for _ in range(3):
        score_list = []
        first_index_list = []
        second_index_list = []
        for i in range(len(num_list)):
            first_index, second_index, score = SSD(image, num_list[i], start, end)
            first_index_list.append(first_index)
            second_index_list.append(second_index)
            score_list.append(score)

        score_list = np.array(score_list)
        first_match = np.argmin(score_list)
        score = score_list[first_match]
        # print(first_match, score)
        image[:,first_index_list[first_match]:second_index_list[first_match]] = 255

        if showGraph:
            plt.plot(range(10),score_list)
            plt.show()
            plt.imshow(image, cmap='gray')
            plt.show()

        if score <= 0.17:
            results.append([first_index_list[first_match],first_match])
            
    results.sort(key=lambda results: results[0])
    results = np.array(results)[:,1].tolist()
    strings = [str(i) for i in results]
    final_score = int("".join(strings))
    return final_score

def evaluate(image, start, end):
    numbers = []
    for i in range(10):
        temp = cv2.imread("{}.png".format(i))
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        numbers.append(temp)
    box = image[start:end,240:1030]
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    ret, masked = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    masked = masked.astype('uint8')
    leftScore = getScores(masked, numbers, 210,280, True)
    rightScore = getScores(masked, numbers, 490,560)
    print(leftScore, rightScore)

if __name__ == "__main__":
    img = cv2.imread("frame001.png")
    evaluate(img, 605, 645)

    # first,second, value= SSD(masked, numbers[6], 200, 300)
    # masked[:, first] = [0]
    # masked[:, second] = [0]
    # plt.imshow(masked, cmap='gray')
    # plt.show()
    # numbers = []
    # for i in range(10):
    #     temp = cv2.imread("{}.png".format(i))
    #     temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    #     numbers.append(temp)

    # box = img[581:621,240:1030]
    # # box = img[605:645, 240:1030]
    # print(box.shape)
    # gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    # ret, masked = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    # masked = masked.astype('uint8')
    # test = getScores(masked, numbers, 210,280)
    # # test = getScores(masked, numbers, 490,560)
    # print(test)

