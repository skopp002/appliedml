from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt



'''Trying linear regression since its easier to imagine'''
model1 = LinearRegression()
colNames = ['temp','nausea','lumbar_pain','freq_urine','micturition_pain','burning','d1_inflammation','d2_nephritis']
data = pd.read_csv("data/acute_inflammation.tsv", sep='\t', header=None)
data.columns = colNames

data.nausea = pd.Categorical(data.nausea, ['yes','no'], ordered = False)
data.nausea = data.nausea.cat.codes
data.lumbar_pain = pd.Categorical(data.lumbar_pain, ['yes','no'])
data.lumbar_pain = data.lumbar_pain.cat.codes
data.freq_urine = pd.Categorical(data.freq_urine, ['yes','no'])
data.freq_urine = data.freq_urine.cat.codes
data.micturition_pain = pd.Categorical(data.micturition_pain, ['yes','no'])
data.micturition_pain = data.micturition_pain.cat.codes
data.burning = pd.Categorical(data.burning, ['yes','no'])
data.burning = data.burning.cat.codes
data.d1_inflammation = pd.Categorical(data.d1_inflammation, ['yes','no'])
data.d1_inflammation = data.d1_inflammation.cat.codes
data.d2_nephritis = pd.Categorical(data.d2_nephritis, ['yes','no'])
data.d2_nephritis = data.d2_nephritis.cat.codes

''' So lets start normalizing temp values '''
temparr = np.array(data['temp'])
tempscaled = preprocessing.scale(temparr)
data.insert(1, "tempscaled", tempscaled,False)

X = data.drop(['temp','d1_inflammation','d2_nephritis'],axis="columns")
print(X)

y1 = data.d1_inflammation

print("X and y1" , X, y1)
y2 = data.d2_nephritis
model1.fit(X,y1)
model1.predict([[1.5,1,1,0,0,1]])

model2 = LinearRegression()
model2.fit(X,y2)
model2.predict([[1.5,1,1,0,0,1]])

model1.score(X,y1)

