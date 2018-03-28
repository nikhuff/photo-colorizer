import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
# from keras.models import Sequential
# from keras.layers import Conv2D, Dense, Activation

def main():
    data_images, target_images = np.load('./assets/data.npy'), np.load('./assets/target.npy')
    colorizer = MLPRegressor()
    print(data_images[0].shape)
    print(target_images[0].shape)
    data = []
    targets = []
    for i, image in enumerate(data_images):
        for row in range(1, len(image) - 1):
            for column in range(1, len(image[row]) - 1):
                kernal = np.zeros((3, 3))
                kernal[0][0] = image[row - 1][column - 1]
                kernal[0][1] = image[row - 1][column]
                kernal[0][2] = image[row - 1][column + 1]
                kernal[1][0] = image[row][column - 1]
                kernal[1][1] = image[row][column]
                kernal[1][2] = image[row][column + 1]
                kernal[2][0] = image[row + 1][column - 1]
                kernal[2][1] = image[row + 1][column]`
                kernal[2][2] = image[row + 1][column + 1]

                data.append(kernal)
                targets.append(target_images[i][row][column])

    colorizer.fit(np.array(data), np.array(targets))
    joblib.dump(colorizer, 'colorizer.pkl')
                
if __name__ == '__main__':
    main()