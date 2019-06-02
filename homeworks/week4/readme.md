Find all the 9s!
1. Find the 9s using K-Nearest neighbours for Minkowski metric of order (1, 2, 3).
Lets first start with understanding what is Minkowski metric:
Typically distance measures in classifiers are :


    Euclidean distance - Square root of sum of squares of differences for every dimension/feature of the datapoints
                         Basically a the length of the straight line connecting 2 points. This is the distance we will use 
                         for Minkowski metric of order 2
                         
    Hamming distance - This is typically used for categorical data points only. Which means the distance between a single feature of the 2 data points is either 0 or 1.
                       With this the hamming distance for 2 data points becomes the number of features that are different.  
   
    Manhattan distance - Sum of differences in every feature. Basically length of lines connecting 2 points with 
                         right angle movements only. This is the metric we will use for Minkowski metric of order 1
   
    Minkowski distance - Minkowski is a generalization of Euclidean and Manhattan distances. In here, for the order 3, we can visualize 
                         it as the distance between 2 points in 3 dimensional space. i.e along x, y and z axes. We cube the difference for each dimension
                         and take a cube root.
                         
                          

                         
                         
                         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
References:
https://pdfs.semanticscholar.org/b3b4/445cb9a2a55fa5d30a47099335b3f4d85dfb.pdf     