import os
# import cv2
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

# def generate_gray_photos():
#     for filename in os.listdir('./assets'):
#         gray = cv2.imread('./assets/' + filename, 0)
#         cv2.imwrite('./assets/gray' + filename, gray)

def get_urls():
    fh = open('./assets/fall11_urls.txt')
    urls = []
    lines = 1
    for x in range(lines):
        line = fh.readline()
        urls.append(line.split('\t')[1])
    return urls

def predictImageProcessing(image):
    result = []
    paddedImage = np.copy(image)
    paddedImage = np.pad(paddedImage, 1, 'reflect')

    height, width = image.shape
    for i in range(height):
            for j in range(width):
                #append each one to an array
                miniresult = [paddedImage[i][j]  , paddedImage[i+1][j]  , paddedImage[i+2][j],
                paddedImage[i][j+1], paddedImage[i+1][j+1], paddedImage[i+2][j+1],
                paddedImage[i][j+2], paddedImage[i+1][j+2], paddedImage[i+2][j+2]]
                #append each set of surrounding pixels to an array
                result.append(miniresult)

    return np.array(result)

def rgbArrayToImage(rgbArray, width, height):
    return rgbArray.reshape(width, height, 3)

def preprocessTrainGray(allImages):

    result = []

    for image in allImages:

        #pad the image by one pixel
        paddedImage = np.copy(image)
        paddedImage = np.pad(paddedImage, 1, 'reflect')

        height, width = image.shape

        #loop through each pixel and extract surrounding pixels
        for i in range(height):
            for j in range(width):
                #append each one to an array
                miniresult = [paddedImage[i][j]  , paddedImage[i+1][j]  , paddedImage[i+2][j],
                paddedImage[i][j+1], paddedImage[i+1][j+1], paddedImage[i+2][j+1],
                paddedImage[i][j+2], paddedImage[i+1][j+2], paddedImage[i+2][j+2]]
                #append each set of surrounding pixels to an array
                result.append(miniresult)

    return np.array(result)

def preprocessColor(allImages):
    result = []

    numPixels = 0
    for image in allImages:
        height, width, channel = image.shape
        reshape = image.reshape(height * width, channel)
        result.append(reshape)
        numPixels += (height * width)
    
    return np.array(result).reshape(numPixels, 3)

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

    colorResult = preprocessColor(color)
    grayResult = preprocessTrainGray(grey)

    return grayResult, colorResult

def main():
    data, target = create_dataset()
    print(data.shape)
    print(target.shape)
    np.save("./assets/data", data)
    np.save("./assets/target", target)

if __name__ == '__main__':
    main()