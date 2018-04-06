import numpy as np
import cv2
from sklearn.neural_network import MLPRegressor
from photo_processing import predictImageProcessing, rgbArrayToImage
from sklearn.externals import joblib
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Activation
# from keras.wrappers.scikit_learn import KerasRegressor

def model():
    model = Sequential()
    model.add(Dense(10, input_shape=(2,)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, 3, input_shape=(471, 500, 3)))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def main():
    grey_pixels, color_pixels = np.load('./assets/data.npy'), np.load('./assets/target.npy')
    print(grey_pixels.shape, color_pixels.shape)
    # print(data_images[0].shape, target_images[0].shape)
    # regressor = KerasRegressor(build_fn=model, nb_epoch=100)
    # regressor.fit(data_images, target_images)

    # regressor.model.save('./assets/colorizer.h5')
    algorithm = MLPRegressor()
    model = algorithm.fit(grey_pixels, color_pixels)
    joblib.dump(model, 'colorizerModel.pkl')

    testImage = cv2.imread('./assets/testImage.png', 0)
    width, height = testImage.shape
    testArray = predictImageProcessing(testImage)
    rgbArray = model.predict(testArray)
    predictedImage = rgbArrayToImage(rgbArray, width, height)
    cv2.imwrite('./assets/predictedImage.png', predictedImage)
                
if __name__ == '__main__':
    main()
