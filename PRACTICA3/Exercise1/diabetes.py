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
# READ THE diabetes.csv #
df = pd.read_csv('diabetes.csv') # File in the directory's path
# Minoritary and mayoritary class
print(df['Outcome'].value_counts())
# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
## Split train-test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)
## SVC classifier ###
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print("SVC classifier METRICS")
print("\tHamming Loss: ", metrics.hamming_loss(y_test, prediction))
print("\tAccuracy Score: ", metrics.accuracy_score(y_test, prediction))
print("\tF1 score micro: ", metrics.f1_score(y_test,prediction,average='micro'))
print("\tError: ", 1-metrics.accuracy_score(y_test, prediction))
# Random Forest classifier
model2.fit(X_train, y_train)
prediction = model2.predict(X_test)
print("Random Forest Classifier metrics")
print("\tHamming Loss: ", metrics.hamming_loss(y_test, prediction))
print("\tAccuracy Score: ", metrics.accuracy_score(y_test, prediction))
print("\tF1 score micro: ", metrics.f1_score(y_test,prediction,average='micro'))
print("\tError: ", 1-metrics.accuracy_score(y_test, prediction))
