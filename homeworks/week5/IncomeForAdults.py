#Problem statement - https://abtinshahidi.github.io/files/week5.pdf
# This is an assignment to find the income of adults dataset
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

'''Observations from the data:
 After cleaning the rows with missing data, our dataset has 30162 rows which is 32561 which is 93% of the data
 For remaining ratios we will consider 30162 as 100%
 Workclass = Private  is 75% of the data
 
'''


class DataModel:
    data = pd.DataFrame
    def __init__(self,filename):
        colnames = ['age', 'workclass', 'finalweight', 'education', 'eduYrs', 'maritalstatus', 'occupation',
                    'relationship', 'race', 'sex', 'capitalgain', 'captialloss', 'hoursperweek', 'nationality',
                    'incomerange']
        data = pd.read_csv(filename, sep=',', header=None)
        data.columns = colnames
        #data.workclass = pd.Categorical(data.workclass, ['Federal-gov', 'Local-gov','Never-worked','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay'], ordered=False)
        #data.workclass = data.workclass.cat.codes
        #About 6-10% data has missing values. It seems ok to lose the data points in such cases since it is much smaller portion of entire dataset
        df = data.replace('[?]', np.NaN, regex=True).dropna(subset = ['workclass', 'occupation', 'nationality'])
        label_encoder = preprocessing.LabelEncoder()
        df['workclass'] = label_encoder.fit_transform(df['workclass'])
        df.maritalstatus = label_encoder.fit_transform(df.maritalstatus)
        df.occupation = label_encoder.fit_transform(df.occupation)
        df.relationship = label_encoder.fit_transform(df.relationship)
        df.race = label_encoder.fit_transform(df.race)
        df.sex = label_encoder.fit_transform(df.sex)
        df.nationality = label_encoder.fit_transform(df.nationality)
        df.age = preprocessing.scale(df.age)
        df.finalweight = preprocessing.scale(df.finalweight)
        df.capitalgain = preprocessing.scale(df.capitalgain)
        #df.capitalloss = preprocessing.scale(df.capitalloss)
        df.hoursperweek = preprocessing.scale(df.hoursperweek)
        df.incomerange = pd.Categorical(df.incomerange,[' <=50K ',' >50K'], ordered=True)
        df.incomerange = df.incomerange.astype("category").cat.codes
        # ord = OrdinalEncoder()
        # ord.transform([[' preschool',0],[' 1st-4th',1],[' 5th-6th',2],[' 7th-8th',3],[' 9th',4],[' 10th',5],[' 11th',6],[' 12th',7],[' HS-grad',8],[' Some-college',9], [' Prof-school',10],[' Bachelors',11], [' Assoc-acdm',12],[' Assoc-voc',13],[' Masters',14],[' Doctorate',15]])
        # ord.fit_transform(df['education'])
        df.education = pd.Categorical(df.education, [' preschool',' 1st-4th',' 5th-6th',' 7th-8th',' 9th',' 10th',
                                                     ' 11th',' 12th',' HS-grad',' Some-college',' Prof-school',' Bachelors',' Assoc-acdm'
                                                     ,' Assoc-voc',' Masters',' Doctorate'], ordered=True)
        #eduYrs came pretty close to education. Hence dropping education
        df.education = df.education.astype("category").cat.codes

        df = df.drop(['education','hoursperweek'], axis="columns")
        df.finalweight = preprocessing.scale(df.finalweight)
        # df.set_index("incomerange", inplace=True)
        # df1 = df.loc(df['incomerange'] == 0)
        # df2 = df.loc(df['incomerange'] == 1)
        # plt.scatter(df1['eduYrs'], df1['race'], color='green', marker='o')
        # plt.scatter(df2['eduYrs'], df2['race'], color='red', marker='x')
        # plt.xlabel("education years")
        # plt.ylabel("race")
        # plt.show()
        self.data = df

    def exploreDataset(self):
         df1 = self.data(self.data['incomerange'] == 0)
         df2 = self.data(self.data['incomerange'] == 1)
         plt.scatter(df1['eduYrs'],df1['race'], color='green',marker='o' )
         plt.scatter(df2['eduYrs'], df2['race'], color='green', marker='o')
         plt.xlabel("education years")
         plt.ylabel("race")
         plt.show()



if __name__ == '__main__':
    d = DataModel("/Users/sunitakoppar/PycharmProjects/datasets/adult/adult.data")
    X = d.data.drop(['incomerange'], axis="columns")
    y = d.data.incomerange
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    rndClf = RandomForestClassifier(n_estimators=100)
    rndClf.fit(X_train,y_train)
    y_rnd_pred = rndClf.predict(X_test)
    print("Random Forest accuracy ", accuracy_score(y_test,y_rnd_pred))
    kernelmethod = 'rbf'
    svclassifier = SVC(kernel=kernelmethod,gamma='auto')  #rbf  #sigmoid  #poly
    svclassifier.fit(X_train, y_train)
    y_svm_pred = svclassifier.predict(X_test)
    print("SVM ",kernelmethod," kernel ", accuracy_score(y_test,y_svm_pred))
    x = np.linspace(2.5,-1, 6033)
    y = np.linspace(2.5,-1, 6033)
    colors=[]
    for i in y_svm_pred:
        if i == 1:
            colors.append("r")
        else:
            colors.append("b")
    np.random.seed(200)
    x1 = 1.2 + 0.5 * np.random.randn(2000)
    y1 = 1.5 + 0.2 * np.random.randn(2000)
    x2 = 0 + 0.3 * np.random.randn(2000)
    y2 = 0 + 0.4 * np.random.randn(2000)
    #fig = plt.scatter(x, y, c=colors, s=0.5) #Couldnt get this line to plot correctly.
    #Hence leaving it commented
    plt.plot(x1, y1, "r.", markersize=10)
    plt.plot(x2, y2, "b.", markersize=10)
    plt.show()

'''     
Old school way of converting

df['workclass'] = pd.DataFrame({'workclass': ['Federal-gov', 'Local-gov','Never-worked','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay']})
        df.workclass = df.workclass.astype("category").cat.codes
        df.maritalstatus = pd.DataFrame({'maritalstatus': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                                                          'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']})
        df.maritalstatus = df.maritalstatus.astype("category").cat.codes
        df.occupation = df.occupation({'occupation':['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                           'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']})
        df.occupation = df.occupation.astype("category").cat.codes
        df.relationship = df.relationship({'relationship':['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']})
        df.relationship =  df.relationship.astype("category").cat.codes
        df.race = df.race({'race':['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']})
        df.race = df.race.astype("category").cat.nodes
        df.sex = df.sex({'sex':['Male','Female']})
        df.sex = df.sex.astype("category").cat.codes
        df.nationality = df.nationality({'nationality': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                                                         'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China',
                                                         'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
                                                         'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                                                         'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                                                         'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']})
        df.nationality = df.nationality.astype("category").cat.codes'''