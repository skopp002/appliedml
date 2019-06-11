import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as  tf

class simpleNN:
    dataset = pd.DataFrame
    X = pd.DataFrame
    y = pd.DataFrame
    cnn_model = Sequential()

    def __init__(self, filename):
        data = pd.read_csv(filename, sep=',', header=None)
        colnames = ['x1','x2','color']
        data.columns = colnames
        label_encoder = preprocessing.LabelEncoder()
        data.color = label_encoder.fit_transform(data.color)
        self.X = data.drop(['color'], axis="columns")
        self.y = data.color
        self.dataset = data

    def builtInCNN(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20)
        self.cnn_model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])
        # x_train_norm = tf.utils.keras.normalize(X_train, axis=1)
        # y_train_norm = tf.utils.keras.normalize(y_train, axis =1)
        Xarr = np.asarray(X_train)
        yarr = np.asarray(y_train)
        self.cnn_model.add(Dense(30, input_dim=2, activation='relu'))
        self.cnn_model.add(Dense(3, activation='relu'))
        self.cnn_model.fit(Xarr, yarr)
        print(self.cnn_model.summary())
        test_eval = self.cnn_model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])


def sigmoid(x):
        """Sigmoid function activation"""
        print("Printing x in sigmoid ", x)
        #wx = []
        # for x in Xarr:
        from numpy import exp
        return ( 1 / (1 + exp(-x)))
             #wx.append( 1 / (1 + exp(-x)))
        # return wx



def sigmoid_derivative(value):
    return value * (1 - value)


def rectified_LU(x):
    """
    Rectified Linear Unit (ReLU) activation
    """
    return max(0, x)

def rectified_LU_derivative(value):
    if value > 0:
        return 1
    else:
        return 0




if __name__ == '__main__':
    snn = simpleNN("nndata.csv")
    X_train, X_test, y_train, y_test = train_test_split(snn.X, snn.y, test_size=0.20)
    weights = np.random.random((2,1)) - 1
    Xarr = np.asarray(X_train)
    yarr = np.asarray(y_train)
    adjustedWeights = []
    for i in range(len(Xarr)):
        # Each element x here is a 1 x 2 matrix of features. We multiply this with a 2 x 1 weight vector.
        # This should give us a scalar value.
        output = rectified_LU(np.dot(Xarr[i],weights))
        error = yarr[i] - output
        adjust = error * rectified_LU_derivative(output)
        adjustedWeights.append(adjust)

    weights = np.dot(Xarr.T, adjustedWeights)
    print("Weights we computed are ", weights)
    print("Now lets try using the built in classifier")
    snn.builtInCNN()


