import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import struct
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import math
from sklearn.neighbors import NearestNeighbors

class Digit:
    def __init__(self):
        self.label = ''
        self.data = []

    def distance(self, digit):
        sum = 0
        for i in range(0, len(self.data)):
            sum = sum + (self.data[i] - digit.data[i]) ** 2
        return sum ** .5

# minkowski order 2
def euclidean_distance(X, Y):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(X, Y)))

# minkowski order n
def minkowski_distance(X, Y, order=3):
        return sum((x - y) ** order for x, y in zip(X, Y)) ** 1 / order

# minkowski order 1
def manhattan_distance(X, Y):
        return sum(abs(x - y) for x, y in zip(X, Y))

def readIdxFiles(filename):
    with open(filename,'rb')as f:
        zero, data, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I',f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(),dtype=np.uint8).reshape(shape)

def loadMnistFromFiles():
    raw_train = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/train-images-idx3-ubyte")
    #here we are flattening every 28 x 28 2-dimensional array into 1 x (28*28) 1 dimension array
    X_train = np.reshape(raw_train, (60000,28*28))
    Y_train = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/train-labels-idx1-ubyte")
    raw_test = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/t10k-images-idx3-ubyte")
    #test data has 10000 records
    X_test = np.reshape(raw_test,(10000,28*28))
    Y_test = readIdxFiles("/Users/sunitakoppar/PycharmProjects/datasets/t10k-labels-idx1-ubyte")
    return X_train,X_test,Y_train,Y_test






if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadMnistFromFiles()
    print("data shape is ", X_train)
    print("target is ",y_train)

    #Method 1 of reading the data files
    #Lets explore the dataset
    #loadMnistFromFiles()
    #This shows the dimensions of the file. (60000, 28, 28)
    #the output indicates that there are 60000 records with each record being a 28 * 28 matrix.