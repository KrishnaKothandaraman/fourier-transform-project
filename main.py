import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Images/homer.jpeg")
savingPath = "/Users/krishnakothandaraman/PycharmProjects/backgroundRemovalFromImage/processedImages"

# for turning background white for easier detection
segmenter = SelfiSegmentation()
imgOut = segmenter.removeBG(img, (255, 255, 255), threshold=0.5)

# convert image to gray scale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = gray_img.shape

# white_padding = np.zeros((50, width, 3))
# white_padding[:, :] = [255, 255, 255]
# rgb_img = np.row_stack((white_padding, img))

gray_img = 255 - gray_img
gray_img[gray_img > 100] = 255
gray_img[gray_img <= 100] = 0
# black_padding = np.zeros((50, width))
# gray_img = np.row_stack((black_padding, gray_img))

kernel = np.ones((30, 30), np.uint8)

# fill inside of image
filledImage = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)
filledImageNormalized = np.uint8(filledImage)

edges = cv2.Canny(filledImageNormalized, 100, 200)

title = ['edges']
images = [edges]

for i in range(1):
    plt.subplot(1, 1, i + 1), plt.imshow(images[i], 'gray')
    plt.title(title[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
