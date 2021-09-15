import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

img = cv2.imread("Images/homer.jpeg")
savingPath = "/Users/krishnakothandaraman/PycharmProjects/backgroundRemovalFromImage/processedImages"
segmenter = SelfiSegmentation()

imgOut = segmenter.removeBG(img, (255, 255, 255), threshold=0.5)

cv2.imwrite(os.path.join(savingPath,'processed.jpg'), imgOut)

print("done")