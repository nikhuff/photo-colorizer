import os
import cv2
import numpy as np
import scipy
from skimage import io
from skimage.transform import resize

def average(pixel):
    return np.average(pixel)

def color_to_grayscale(image):
    grey = np.zeros((image.shape[0], image.shape[1]))
    for row in range(len(image)):
        for column in range(len(image[row])):
            grey[row][column] = average(image[row][column])
    return grey

def resize_image(image):
    return resize(image, (256, 256))

def url_to_image(url):
    try:
        image = io.imread(url)
    except Exception:
        image = np.zeros((256, 256, 3))
    return image

def generate_gray_photos():
    for filename in os.listdir('./assets'):
        gray = cv2.imread('./assets/' + filename, 0)
        cv2.imwrite('./assets/gray' + filename, gray)

def get_urls():
    fh = open('./assets/fall11_urls.txt')
    urls = []
    lines = 1
    for x in range(lines):
        line = fh.readline()
        urls.append(line.split('\t')[1])
    return urls

def create_dataset():
    grey = []
    color = []
    # use for urls
    urls = get_urls()
    for url in urls:
        image = url_to_image(url)
        # image = resize_image(image)
        color.append(image)
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
    data, target = create_dataset()
    np.save("./assets/data", data)
    np.save("./assets/target", target)

if __name__ == '__main__':
    main()