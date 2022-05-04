#!  /bin/python
# Author: Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
## Classifiers
model=SVC()
model2=RandomForestClassifier(n_estimators=25)
# READ THE BREAST CANCER DATASET #
print("### breast-cancer-wisconsin dataset ###")
cols_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion'
,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv('breast-cancer-wisconsin.data', header = None, names = cols_names, sep = ",")
# Minoritary and mayoritary class -> class = 2: Mayoritary; Class = 4: Minoritary
print(df['Class'].value_counts())
# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
## Split train-test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
## SVC classifier ###
model.fit(X_train, y_train)
