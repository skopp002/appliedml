import numpy as np
import matplotlib.pyplot as plt

def getScaledTemp(actual):
    return (float(actual) - 38)

def getEncoded(actual):
    if(actual == 'yes'):
        return 1
    elif(actual == 'no'):
        return 0

def loadDataIntoMatrix(filename):
    # p represents predictors and d represents the decision or the output variable or label as in some references.
    p = []; d1 = []; d2 = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        '''We know from eyeballing this data (specifically corelation between d2 and temperature
        that 38 can be taken as mean and anything below it can be taken as below normal and anything above can be taken as 
        above normal temperature. Hence we will scale temperature based on this assumption. For rest of the categorical 
        values, there are multiple encoders available, however I plan to vary them between 0 and 1 or 1 and -1 etc. Hence 
        will define a custom encoder essentially a simple function
        '''
        temperature = getScaledTemp(lineArr[0])
        nausea = getEncoded(lineArr[1])
        lumbarpain = getEncoded(lineArr[2])
        freq_urine = getEncoded(lineArr[3])
        micturitionpain = getEncoded(lineArr[4])
        burning = getEncoded(lineArr[5])
        bladderinflammation = getEncoded(lineArr[6])
        nephritis = getEncoded(lineArr[7])
        # This is our data matrix
        p.append([temperature, nausea, lumbarpain, freq_urine, micturitionpain, burning])
        #These are our decision matrices
        d1.append(bladderinflammation)
        d2.append(nephritis)

    return  p,d1,d2


#Sigmoid function to return binary class values (0 or 1) based on input Matrix
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))


def gradientAscent(p,d, alpha, iters):
    #In out case, this will be a 90 * 5 matrix. We will use 25% of the data for test
    data = np.mat(p)
    # The labels are 90 * 1 matrix. In order to multiply,
    label = np.mat(d).transpose()

    #Returns the dimensions of the input data matrix
    m,n = np.shape(data)
    # We start out with a matrix with all coefficients as 1.
    # The shape of weights is such that it can be multiplied with data
    weights = np.ones((n,1))
    for k in range(iters):
        h = sigmoid(data * weights)
        # The deviation between predicted and given value
        err = (label - h)
        weights = weights + alpha * data.transpose()*err
    return weights


def plotBestFit(weights, dataMat, labelMat):
    # Returns the matrix as n dimensional array
    w = weights.getA()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    x1 = []; y1 = []
    x2 = [];y2 = []
    for i in range(n):
        if int(labelMat[i] == 1):
            # If
            x1.append(dataArr[i,1]);y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1]);y2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1,y1, s=30, c='red', marker='s')
    ax.scatter(x2,y2, s=30, c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x,y)
    plt.xlabel("Symptoms");plt.ylabel("Disease");
    plt.show()

if __name__ == '__main__':
    ALPHA = 0.001
    NUM_OF_ITERATIONS = 500
    data,label_d1, label_d2 = loadDataIntoMatrix("data/acute_inflammation.tsv")
    weightsMat = gradientAscent(data,label_d1,ALPHA,NUM_OF_ITERATIONS)
    plotBestFit(weightsMat,data,label_d1)
    #weightsMat = gradientAscent(data,label_d2,ALPHA,NUM_OF_ITERATIONS)
    #plotBestFit(weightsMat,data,label_d2)