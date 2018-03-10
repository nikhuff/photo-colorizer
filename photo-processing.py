import os
import cv2

for filename in os.listdir('./assets'):
    gray = cv2.imread('./assets/' + filename, 0)
    cv2.imwrite('./assets/gray' + filename, gray)