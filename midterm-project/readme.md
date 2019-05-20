### MidTerm Project Writeup


The dataset we are dealing with has 1 continuous and 5 categorical variables. 
We typically start out with exploring the data and trying to understand corelations.
Slicing and dicing the data using python packages like pandas is shown in  ./ExploreDataSet.py
And slicing and dicing using excel is shown in the SunitaKoppar_phys243_DataAnalysis.xlsx

We can assume some model that temperature (fever) is certainly an indication of inflammation. However, 
it could be an inflammation of any part of the body. Given that the d1 and d2 are specific to Urinary bladder 
or Kidney inflammation, we will not be assuming any specific weights before exploring the dataset.

After initial exploration of a single variable dependency on output classes, (Refer to DataAnalysis sheet in the xls) 
we can see some interesting corelations:
1. nausea seems to have some covariance with fever
2. temperature seems to be a clear indicator of d2. Higher the temperature the more likely it is to have nephritis
3. A confirmation of 1 above, nausea seems to have a corelation with d2 as well. 
Nausea may or may not a causative. It could be temperature alone driving the decision for d2. 
4. Lumbar pain as per this data set seems to vary inversely with d2. 
5. There is no clear corelation between d1 and d2

In the MiscCharts sheet, there are additional charts but they do not provide any clear finding, hence we do not use them.
During the exploration, temperature being the only continuous variable caused some skews in the plots. Hence temperature 
had to be scaled. Also, yes and no are not in machine friendly formats and hence had to be coded into numbers, with 1 being
a yes and 0 being a no. 

The problem we have at hand is that of classfication. And we will be using supervised machine learning methods.
The options we have to come up with classification are:
1. Logistic regression
2. Decision Trees / Random Forest
3. Baysean classifier
4. Support Vector Machines

We will be using Logistic Regression in this project. We started with optimization technique of gradient ascent (Code in RejectedApproaches/logisticRegClassifier) 
and realized that it is not suitable for categorical data.
Due to lack of numerical variables, plotting is not very useful on this data. However, some attempt has been made
to plot just for exploration in logisticRegClassifier::plotBestFit method

The final implementation is available in **midterm-project/LogisticRegression.py**
The scores of the models seem to be 1.0 after 40% split (60% training and 40% test) however, even with 25% split between training and test data, there were some cases with random tests which were not 
matching predicting correctly.
With 80/20 split, the prediction on random tests improved and matched the expected results in every random test.

#### If we change the ratio between training and test to be 50:50, the score of the model drops:
Output below:
`_runfile('/Users/sunitakoppar/PycharmProjects/appliedml/midterm-project/LogisticRegression.py', wdir='/Users/sunitakoppar/PycharmProjects/appliedml/midterm-project')
Model trained for  BladderInflammation
Model Prediction for  BladderInflammation  is  [0 1 1 0 1 1 1 0 0 0 0 1 1 1 1 1 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 1 1 0 1 0
 1 1 1 1 0 1 1 0 0 0 1 1 0 0 0 1 0 1 0 1 1 1 0]
Model score for  BladderInflammation  is  1.0
Prediction for random test for  BladderInflammation  is  yes_`

`_Model trained for  Nephritis
Model Prediction for  Nephritis  is  [1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 0 1 1 1 0 1
 0 0 0 0 0 1 0 1 1 0 1 0 0 1 0 1 1 1 1 1 1 1 1]
Model score for  Nephritis  is  0.9666666666666667
Prediction for random test for  Nephritis  is  yes_`
-------------

There are 2 random tests for both output variables with outputs of both classes and the model can be tested by altering the values in 
        symptoms.loc[0] = [36, "no", "no", "no", "yes", "no"]






