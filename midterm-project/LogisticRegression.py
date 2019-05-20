import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class PredictDisease:
    name=""
    predictors=pd.DataFrame
    output=pd.Categorical
    X_train = pd.DataFrame ; X_test = pd.DataFrame
    Y_train = pd.Series ; Y_test = pd.Series
    model = LogisticRegression(solver='lbfgs')

    def __init__(self, predictors, output, name):
        self.predictors = predictors
        self.output = output
        self.name = name
        self.build()

    def trainModel(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.predictors, self.output, test_size=0.4, random_state=0)
        self.model.fit(self.X_train, self.Y_train)
        print("Model trained for ", self.name)

    def testModel(self):
        Y_prediction = self.model.predict(self.X_test)
        print("Model Prediction for ", self.name ," is ", Y_prediction)

    def scoreModel(self):
        print("Model score for ",self.name ," is ", self.model.score(self.X_test, self.Y_test))

    def build(self):
        self.trainModel()
        self.testModel()
        self.scoreModel()

    def getScaledTemp(self, actual):
        return (38 - float(actual))

    def getCategoricalTemp(self,actual):
        if ((35 <= actual).bool or (actual <= 38).bool):
            return 0
        if ((38.1 <= actual).bool):
            return 1


    def randomTest(self,symptoms):
        symptoms['temperature'] = self.getCategoricalTemp(symptoms['temperature'])
        symptoms['nausea'] = symptoms['nausea'].map({'yes': 1, 'no': 0})
        symptoms['lumbar_pain'] = symptoms['lumbar_pain'].map({'yes': 1, 'no': 0})
        symptoms['freq_urine'] = symptoms['freq_urine'].map({'yes': 1, 'no': 0})
        symptoms['micturition_pain'] = symptoms['micturition_pain'].map({'yes': 1, 'no': 0})
        symptoms['burning'] = symptoms['burning'].map({'yes': 1, 'no': 0})
        prediction = self.model.predict(symptoms)
        pred = "yes"
        if prediction[0] == 1:
            pred  = "yes"
        elif prediction[0] == 0:
            pred = "no"
        print("Prediction for random test for ", self.name," is ", pred)


def loadData(filename):
    '''Tried getDummies for this encoding but was not sure the encoding was consistent. Hence did it in an elaborate way'''
    colNames = ['temp', 'nausea', 'lumbar_pain', 'freq_urine', 'micturition_pain', 'burning', 'd1_inflammation',
                'd2_nephritis']
    data = pd.read_csv(filename, sep='\t', header=None)
    data.columns = colNames
    data.nausea = pd.Categorical(data.nausea, ['yes', 'no'], ordered=False)
    data.nausea = data.nausea.cat.codes
    data.lumbar_pain = pd.Categorical(data.lumbar_pain, ['yes', 'no'])
    data.lumbar_pain = data.lumbar_pain.cat.codes
    data.freq_urine = pd.Categorical(data.freq_urine, ['yes', 'no'])
    data.freq_urine = data.freq_urine.cat.codes
    data.micturition_pain = pd.Categorical(data.micturition_pain, ['yes', 'no'])
    data.micturition_pain = data.micturition_pain.cat.codes
    data.burning = pd.Categorical(data.burning, ['yes', 'no'])
    data.burning = data.burning.cat.codes
    data.d1_inflammation = pd.Categorical(data.d1_inflammation, ['yes', 'no'])
    data.d1_inflammation = data.d1_inflammation.cat.codes
    data.d2_nephritis = pd.Categorical(data.d2_nephritis, ['yes', 'no'])
    data.d2_nephritis = data.d2_nephritis.cat.codes
    ''' Temperature showed skewing predictions  during data analysis, so lets start normalizing temp values '''
    temparr = np.array(data['temp'])
    tempscaled = preprocessing.scale(temparr)
    data.insert(1, "tempscaled", tempscaled, False)
    X = data.drop(['temp', 'd1_inflammation', 'd2_nephritis'], axis="columns")
    y1 = data.d1_inflammation
    y2 = data.d2_nephritis
    return X, y1, y2


if __name__ == '__main__':
    X,y1,y2 = loadData("data/acute_inflammation.tsv")
    bladderInflammationPredictor = PredictDisease(X, y1,"BladderInflammation")
    symptoms=pd.DataFrame(columns=['temperature','nausea','lumbar_pain','freq_urine','micturition_pain','burning'])
    symptoms.loc[0] = [36, "no","no", "yes", "yes", "yes"]
    bladderInflammationPredictor.randomTest(symptoms)
    symptoms.loc[0] = [36, "yes", "yes", "no", "yes", "no"]
    bladderInflammationPredictor.randomTest(symptoms)
    print("------------------------------------------------------")
    nephritisPredictor = PredictDisease(X, y2, "Nephritis")
    symptoms.loc[0] = [39, "yes", "no", "yes", "no", "yes"]
    nephritisPredictor.randomTest(symptoms)
    symptoms.loc[0] = [39, "no", "no", "yes", "no", "yes"]
    nephritisPredictor.randomTest(symptoms)















### Scratch pad
# print("Now lets split and observe with training and test set")
#
#
# print("With the training set, prediction for bladder inflammation is ", y1_pred, " with score ", model1.score(X_test,y1_test))
# X_train,X_test,y2_train,y2_test=train_test_split(X,y2,test_size=0.25,random_state=0)
# model2 = LogisticRegression(solver='lbfgs')#liblinear')
# model2.fit(X_train,y2_train)
# y2_pred = model2.predict(X_test)
# print("With the training set, prediction for nephritis is ", y2_pred, " with score ", model2.score(X_test,y2_test))
# d1_prediction = model1.predict([[0,0,0,0,1,0]]) # Toggling the 5th element toggles the output
# d2_prediction = model2.predict([[0,1,1,0,0,1]]) # Toggling the 3rd element toggles the output
