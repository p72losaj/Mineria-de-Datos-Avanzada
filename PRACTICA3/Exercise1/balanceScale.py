#!  /bin/python
# Author: Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# Classifiers
model=SVC()
model2=RandomForestClassifier(n_estimators=25)

# READ THE balance_scale
cols_names = ['Class','Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']
df = pd.read_csv('balance-scale.data', header = None, names = cols_names, sep = ",")
# Transform into binary classification
df['Class'] = [1 if b=='B' else 0 for b in df.Class]
# Minoritary and mayoritary class
print(pd.value_counts(df['Class'], sort = True))
# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
## Split train-test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30)

#######################################################################################
# SVC classifier
model.fit(X_train, y_train)
prediction = model.predict(X_test)
# Calculate the SVC's metrics
print("SVC classifier metrics")
print("\tHamming Loss: ", metrics.hamming_loss(y_test, prediction))
print("\tAccuracy Score: ", metrics.accuracy_score(y_test, prediction))
print("\tF1 score micro: ", metrics.f1_score(y_test,prediction,average='micro'))
print("\tError: ", 1-metrics.accuracy_score(y_test, prediction))
#######################################################################################
# Random Forest classifier
model2.fit(X_train, y_train)
prediction = model2.predict(X_test)
# Calculate de Random Forest's metrics
print("Random Forest Classifier metrics")
print("\tHamming Loss: ", metrics.hamming_loss(y_test, prediction))
print("\tAccuracy Score: ", metrics.accuracy_score(y_test, prediction))
print("\tF1 score micro: ", metrics.f1_score(y_test,prediction,average='micro'))
print("\tError: ", 1-metrics.accuracy_score(y_test, prediction))

