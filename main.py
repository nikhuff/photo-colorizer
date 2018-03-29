import numpy as np
# from sklearn.neural_network import MLPRegressor
# from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor

def model():
    model = Sequential()
    model.add(Dense(32, input_shape=(471,)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, input_shape=(471, 500, 3)))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def main():
    data_images, target_images = np.load('./assets/data.npy'), np.load('./assets/target.npy')
    print(data_images[0].shape, target_images[0].shape)
    regressor = KerasRegressor(build_fn=model, nb_epoch=100)
    regressor.fit(data_images[0], target_images)

    regressor.model.save('./assets/colorizer.h5')
                
if __name__ == '__main__':
    main()
