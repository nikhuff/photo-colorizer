import os
import cv2
import numpy as np

def generate_gray_photos():
    for filename in os.listdir('./assets'):
        gray = cv2.imread('./assets/' + filename, 0)
        cv2.imwrite('./assets/gray' + filename, gray)

def create_dataset():
    gray = []
    color = []
    for filename in os.listdir('./assets/color'):
        cimg = cv2.imread('./assets/color/' + filename)
        gimg = cv2.imread('./assets/gray/gray' + filename)
        gray.append(gimg)
        color.append(cimg)

    return np.array(gray), np.array(color)

def main():
    train, test = create_dataset()

if __name__ == '__main__':
    main()