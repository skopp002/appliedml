import numpy as np
import matplotlib.pyplot as plt

#Lets take Q ~ X^2
# In the example, we have 2 arrays
#First array is Array of all Mu's.

def calculateChiSq(m,s):
    mu = np.array(m)
    sig = np.array(s)
    mean = mu.size
    stdev= np.sqrt(2 * mu.size)
    print("Mean and standard dev for given arrays are ", mean , " and ", stdev)


mu = [10,12,11,15,11,13,16]
sigma = [1,2,1,1.5,3,1,2]
calculateChiSq(mu,sigma)