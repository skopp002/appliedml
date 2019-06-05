import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def loadMnistFromOpenml():
    try:
        #sklearn provides some open datasets which can be pulled based on dataset id
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8)
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
        #The loading of the file was taking too long. Hence trimmed the dataset to use first 5000 records only
    return  train_test_split(mnist["data"][:5000], mnist["target"][:5000], test_size=0.2)


# minkowski order 2
def euclidean_distance(X, Y):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(X, Y)))

# minkowski order n
def minkowski_distance(X, Y, order=3):
        return sum((x - y) ** order for x, y in zip(X, Y)) ** 1 / order

# minkowski order 1
def manhattan_distance(X, Y):
        return sum(abs(x - y) for x, y in zip(X, Y))

def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #Encountered overflow error with Minkowski of order 3
        dist = euclidean_distance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
        import operator
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadMnistFromOpenml()
    print("data shape is ", X_train)
    print("target is ",y_train)
    neighbors=[]
    #for img in X_train:
     #   neighbor = "" #getKNeighbors(X_train, img, 10)  --> comment the invocation to avoid too much computation
      #  neighbors.append(neighbor)

    print("Non labeled clusters ", neighbors)


    #For Manhattan distance, use order = 1, for Euclidean use 2 and for Minkowski of order 3, pass order=3
    clf1 = KNeighborsClassifier(n_neighbors=10, p=1).fit(X_train,y_train)
    y1_pred = clf1.predict(X_test)
    print ("KNN Model score with order 1 is " ,clf1.score(X_test, y_test))
    clf2 = KNeighborsClassifier(n_neighbors=10, p=2).fit(X_train, y_train)
    y2_pred = clf2.predict(X_test)
    print("KNN Model score with order 2 is ", clf2.score(X_test, y_test))
    clf3 = KNeighborsClassifier(n_neighbors=10, p=3).fit(X_train, y_train)
    y3_pred = clf3.predict(X_test)
    print("KNN Model score with order 3 is ", clf3.score(X_test, y_test))
    plt.figure(figsize=(12, 12))
    j = 331
    for i in range(0, 9):
        example = X_test[i].reshape(28, 28)
        plt.subplot(j);
        plt.imshow(example, cmap=mpl.cm.binary)
        plt.text(0, 2, "prediction = {}".format(y2_pred[i]))
        j += 1
    plt.show()

    # Find all 9s
    print("All 9s")
    nines_train = X_train[y_train == 9]
    nines_test = X_test[y_test == 9]
    clf4 = KNeighborsClassifier(n_neighbors=10, p=3).fit(nines_train, y_train[y_train == 9])
    y_pred_9 = clf4.predict(nines_test)
    print("KNN Model score for finding 9s is  ", clf3.score(nines_test, y_test[y_test == 9]))
    plt.figure(figsize=(12, 12))
    j = 331
    for i in range(0, 9):
        example = nines_test[i].reshape(28, 28)
        plt.subplot(j);
        plt.imshow(example, cmap=mpl.cm.binary)
        plt.text(0, 2, "prediction = {}".format(y_pred_9[i]))
        j += 1
    plt.show()

    print("Confusion matrix with KNN for only 9s")
    print(confusion_matrix(y_test[y_test == 9], y_pred_9))


    #Lets also look at confusion matrix to see how well the classifier categorized
    print("Confusion matrix with KNN")
    print(confusion_matrix(y_test, y2_pred))  #Lets use the order 2

    #Lets try Decision Tree
    decClf = DecisionTreeClassifier(criterion='entropy', max_depth=5,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=10,
            splitter='random')
    decClf.fit(X_train,y_train)
    dec_pred = decClf.predict(X_test)
    #print("Decision tree prediction ", dec_pred)
    #The higher numbers (non zero) in non-diagonal spots indicates that the predicted labels did not tally with actual labels in all cases
    print("Confusion matrix with decision trees")
    print(confusion_matrix(y_test,dec_pred))

    #Random Forest Classifier
    rndClf = RandomForestClassifier() #n_jobs=2, random_state=0
    rndClf.fit(X_train,y_train)
    rnd_pred = rndClf.predict(X_test)
    print("Confusion matrix with Random Forest")
    print(confusion_matrix(y_test, rnd_pred))

    #As shown in the confusion matrix output, Random Forest has higher number of successful predictions


    #Lets also get the classification reports by all the classifiers:
    print("KNN .......")
    print(classification_report(y_test, y3_pred))
    print("Decision Tree .......")
    print(classification_report(y_test, dec_pred))
    print("RandomForest .......")
    print(classification_report(y_test, rnd_pred))