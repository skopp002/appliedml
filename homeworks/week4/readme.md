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


    KNN Model score with order 1 is  0.904
    KNN Model score with order 2 is  0.924
    KNN Model score with order 3 is  0.929
    
    All 9s
    KNN Model score for finding 9s is   0.9428571428571428
                         
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

Here are the classification reports:


KNN .......
              precision    recall  f1-score   support
           0       0.95      0.98      0.97       101
           1       0.89      0.99      0.94       119
           2       0.97      0.84      0.90        88
           3       0.90      0.92      0.91        98
           4       0.96      0.93      0.94       111
           5       0.92      0.95      0.93        81
           6       0.97      1.00      0.99       113
           7       0.93      0.94      0.94       101
           8       0.93      0.85      0.89        82
           9       0.93      0.92      0.92       106
    accuracy                           0.94      1000
   macro avg       0.94      0.93      0.93      1000
weighted avg       0.94      0.94      0.94      1000


Decision Tree .......
              precision    recall  f1-score   support
           0       0.86      0.83      0.84       101
           1       0.75      0.85      0.80       119
           2       0.66      0.58      0.62        88
           3       0.58      0.63      0.61        98
           4       0.65      0.82      0.73       111
           5       0.49      0.37      0.42        81
           6       0.85      0.59      0.70       113
           7       0.71      0.87      0.78       101
           8       0.60      0.55      0.57        82
           9       0.72      0.72      0.72       106
    accuracy                           0.69      1000
   macro avg       0.69      0.68      0.68      1000
weighted avg       0.70      0.69      0.69      1000


RandomForest .......
              precision    recall  f1-score   support
           0       0.96      0.98      0.97       101
           1       0.92      0.96      0.94       119
           2       0.87      0.85      0.86        88
           3       0.82      0.86      0.84        98
           4       0.85      0.85      0.85       111
           5       0.92      0.90      0.91        81
           6       0.93      0.96      0.94       113
           7       0.89      0.89      0.89       101
           8       0.93      0.80      0.86        82
           9       0.84      0.85      0.85       106
    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000
                         
                         
 The F1 scores which are typically looking at harmonic mean of precision and recall (balanced results between how precise and how
 accurate or complete the results are. A regular mean will behave the same for .10 Precision .90 Recall and a 0.4 precision and 0.6 recall.
 In reality the later is better since its more balanced)
 
    
Difference between binary and multiclass classifier:
Typically mis classification rates in binary classifiers is lower than in multiclass classifier. It is also faster to compute binary classifiers however, 
the downside is there is greater pre-processing required with binary classifiers since the problem needs to be translated into a binary classification 
and all the classes need to be individually tried as separate binary classifiers in order to decipher same information. 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
References:
https://pdfs.semanticscholar.org/b3b4/445cb9a2a55fa5d30a47099335b3f4d85dfb.pdf     