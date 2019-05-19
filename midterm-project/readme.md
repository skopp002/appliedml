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

In the MiscCharts sheet, there are additional charts but they do not provide any clear finding, hence we do not use them.

The problem we have at hand is that of classfication. And we will be using supervised machine learning methods.
The options we have to come up with classification are:
1. Logistic regression
2. Decision Trees / Random Forest
3. Baysean classifier
4. Support Vector Machines

We will be using Logistic Regression in this project. The optimization technique we use will be gradient ascent.






