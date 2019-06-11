The python file corresponding to this readme is IncomeForAdults.py

With Random forest, 
The estimator is the number of trees there are expected to be in the random forest.
Tweaking this value did not improve the accuracy infact degraded it a bit. 

```Random Forest accuracy  0.8552958727001492
SVM linear kernel  0.8209845847836897
SVM  Sigmoid kernel 0.7598209845847836
SVM  rbf  kernel     0.8519807724183657  with gamma='scale' accuracy dropped 0.77
    poly took too long to compute hence could not test it
```
      
The idea of SVM kernels is to use a dot product as a new dimension. It is computationally too 
expensive to project all given features on to another plane. The dot product of the 
feature vectors automatically magnifies the impact of both lower and higher values thus increasing the 
gap between the datapoints belonging to different output classes. 
The various types of kernels (other than linear) 

    
The 2 parameters that we pass over to SVM classifier are:
```1. gamma - higher values of gamma could lead to overfitting. The greater the value of gamma, the farther 
   the points from decision boundary are considered. 
2. C - The gap between the classification segments. In our case, the gap between income <=50K and 
       >50. The way this value will impact is, choosing a higher value for C means we want to avoid 
       misclassification. This translated to reduced distance between the support vectors and decision 
       boundary ``` 
  
      

