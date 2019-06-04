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
                         
                          
Passing various order values (1,2 and 3) to KNN, the model score comes up to be
python FindNines.py shows:


    KNN Model score with order 1 is  0.918
    KNN Model score with order 2 is  0.923
    KNN Model score with order 3 is  0.932
                         
The actual scores vary in every execution however, the scores always increase with order. For instance, minkowski order 1 
always has lesser score than order 2 and that in term has lesser score than order 3.      
To compute unlabelled KNN, I tried to compute without using builtin methods. However, it didnt complete
processing for more than 30 mins. The method FindNines.getKNeighbors() is expected to provide the list
of neighbors for all the digits without the labels. However, the computation time was too long and hence
commenting the invocation to this method. The screenshot of the results for Unsupervised KNN (without labels)
is "UnsupervisedNeighbors.png"                      
                         
The confusion matrix is another measure of how well the classifier worked. For a uniform scale or 
comparison, confusion matrix provides a wholestic view of how all the inputs have been classified. 
Typically along X axis, the actual labels are plotted and along Y access the predicted labels by 
the classifier.                         
                         
                         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
References:
https://pdfs.semanticscholar.org/b3b4/445cb9a2a55fa5d30a47099335b3f4d85dfb.pdf     