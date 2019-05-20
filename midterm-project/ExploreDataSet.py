###Dataset donated by
# J.Czerniak, H.Zarzycki, Application of rough sets in the presumptive
# diagnosis of urinary system diseases, Artifical Inteligence and Security
# in Computing Systems, ACS'2002 9th International Conference Proceedings,
# Kluwer Academic Publishers,2003, pp. 41-51


# Attribute Information:
#
# a1 Temperature of patient { 35C-42C }
# a2 Occurrence of nausea { yes, no }
# a3 Lumbar pain { yes, no }
# a4 Urine pushing (continuous need for urination) { yes, no }
# a5 Micturition pains { yes, no }
# a6 Burning of urethra, itch, swelling of urethra outlet { yes, no }
# d1 decision: Inflammation of urinary bladder { yes, no }
# d2 decision: Nephritis of renal pelvis origin { yes, no }


# The first step in data analysis is understanding the dataset. Visualization and statistical analysis
#plays a big role in understanding the data. Lets consider some options to read and understand our dataset

# In this dataset, out of 6 independent variables, we have
#   1 continuous variable  - a1 or temperature
#   4 binary/discrete variables  (a2 - a5)
#
# 2 dependent binary variables
#   d1 and d2
#
# Since the output variable is binary, this is a classification problem
#


import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt



# data = np.load("data/acute_inflammation.tsv", delimiter = ' ', dtype=[('temp',int),
#                                                                             ('nausea',bool),
#                                                                             ('lumbar_pain',bool),
#                                                                             ('freq_urine',bool),
#                                                                             ('micturition_pain',bool),
#                                                                             ('burning',bool),
#                                                                             ('d1_inflammation',bool),
#                                                                             ('d2_nephritis',bool)])
# Faced errors using np.load. Hence trying pandas

colNames = ['temp','nausea','lumbar_pain','freq_urine','micturition_pain','burning','d1_inflammation','d2_nephritis']
data = pd.read_csv("data/acute_inflammation.tsv", sep='\t', header=None)
data.columns = colNames

'''We can get an idea about individual data elements with below commands'''
#Histogram
# fig = plt.figure(figsize=(18,18))
# ax = fig.gca()
# data.hist(ax = ax, bins=120)
# plt.show()
# or
#data['nausea'].plot.hist(alpha=0.5,bins=20

temparr = np.array(data['temp'])
tempscaled = preprocessing.scale(temparr)
#
print(tempscaled)


'''Trying to get corelation'''
data['nausea'] = data['nausea'].map({'yes': 1, 'no': 0})
# data['d1_inflammation'] = data['d1_inflammation'].str.lower().replace({'yes': 1, 'no': 0})
# data.plot(kind='bar', x='temp', y='d1_inflammation',color='r')
# data.plot(kind='bar', x='nausea', y='d1_inflammation',color='g')
# plt.xlabel('Symptoms')
# plt.ylabel('Disease')
# plt.show()



'''trying to get insights into data
#Lets start by converting the categorical values to numeric'''
data.nausea = pd.Categorical(data.nausea, ['yes','no'], ordered = False)
data.nausea = data.nausea.cat.codes
data.lumbar_pain = pd.Categorical(data.lumbar_pain, ['yes','no'])
data.lumbar_pain = data.lumbar_pain.cat.codes
data.freq_urine = pd.Categorical(data.freq_urine, ['yes','no'])
data.freq_urine = data.freq_urine.cat.codes
data.d1_inflammation = pd.Categorical(data.d1_inflammation, ['yes','no'])
data.d1_inflammation = data.d1_inflammation.cat.codes
''' So lets start normalizing temp values '''
data.insert(1, "tempscaled", tempscaled,False)
plt.plot(data.d1_inflammation,data.tempscaled,'o', color='r')
plt.plot(data.d1_inflammation,data.nausea,'+',color='g')
plt.plot(data.d1_inflammation,data.freq_urine,'^',color='b')
plt.show()
plt.legend(['tempScaled','nausea'])




'''This is simpler and replaces all categorical in one go. 
However whether it would assign 1 to yes or 1 to no is pretty adhoc'''
#df = pd.get_dummies(data)
'''This shows the effect of skew by using Temp. 
With this we can understand the need to normalize the temp values'''

#plt.plot(df.d1_inflammation_yes,df.temp,'o', color='r')
# df.insert(1, "tempscaled", tempscaled,False)
# df.d1_inflammation_yes,df.d1_inflammation_no)
# plt.plot(df.d1_inflammation_yes,df.tempscaled,'o', color='r')
# plt.plot(df.d1_inflammation_yes,df.nausea_yes,'^',color='g')
# # plt.plot(df.d1_inflammation_yes,df.lumbar_pain_yes,'x',color='b')
# # plt.plot(df.d1_inflammation_yes,df.freq_urine_yes,'|',color='y')
# # plt.plot(df.nausea_no,df.d1_inflammation_yes)
# # plt.plot(df.nausea_no,df.d1_inflammation_no)
# plt.show()
# plt.legend(['tempScaled','nausea'])


'''Comparison of pandas with sql - https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html
size = count
'''
data.groupby('d1_inflammation').agg({'nausea': np.size})




