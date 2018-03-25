import os
import cv2
import numpy as np
from skimage import io

def average(pixel):
    return np.average(pixel)

def color_to_grayscale(image):
    grey = np.zeros((image.shape[0], image.shape[1]))
    for row in range(len(image)):
        for column in range(len(image[row])):
            grey[row][column] = average(image[row][column])
    return grey


def url_to_image(url):
    image = io.imread(url)
    return image

def generate_gray_photos():
    for filename in os.listdir('./assets'):
        gray = cv2.imread('./assets/' + filename, 0)
        cv2.imwrite('./assets/gray' + filename, gray)

def create_dataset():
    grey = []
    color = []
    # use for urls
    filename = "./assets/urls.txt"
    urls = np.loadtxt(filename, dtype=str)
    for url in urls:
        color.append(url_to_image(url))
    for image in color:
        grey.append(color_to_grayscale(image))
    # use for physical images in assets
    # for filename in os.listdir('./assets/color'):
    #     cimg = cv2.imread('./assets/color/' + filename)
    #     gimg = cv2.imread('./assets/gray/gray' + filename)
    #     gray.append(gimg)
    #     color.append(cimg)

    return np.array(grey), np.array(color)

def main():
    train, test = create_dataset()
    print(train.shape)
    print(test.shape)

if __name__ == '__main__':
    main()