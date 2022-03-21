# Tratamiento de datos

import numpy as np
import pandas as pd

# Preprocesado y modelado

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn import datasets 

#### LOAD THE IRIS DATASET ###############

cols_names = ['sepal length','sepal width', 'petal length','petal width', 'label']
iris = pd.read_csv('iris.data', header = None, names = cols_names, sep = ",")

# split dataset in features and target variable

feature_cols = ['sepal length','sepal width', 'petal length','petal width']
# Features
x = iris[feature_cols] 
# Target variable
y = iris.label 

# Split dataset into training set and test set -> 70% training and 30% test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# PIPELINE ESTIMATOR

pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))

# Instantiate the bagging classifier

bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,max_features=4,max_samples=100,random_state=1, n_jobs=5)

# Fit the bagging classifier

bgclassifier.fit(X_train, y_train)

# Model scores on test data

test = bgclassifier.score(X_test, y_test)

# Predict the response for test dataset

y_pred = bgclassifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?

accuracy = metrics.accuracy_score(y_test, y_pred)

# Tase error

error = 1 - accuracy

# Show the results

print('##########\nIRIS DATA#####')

print('Test -> ', test); print('Error-> ', error)

### LOAD THE BALANCE-SCALE DATASET #####

cols_names = ['Class','Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']
balance_scale = pd.read_csv('balance-scale.data', header = None, names = cols_names, sep = ",")

# split dataset in features and target variable

feature_cols = ['Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']	
# Features
x = balance_scale[feature_cols] 
# Target variable
y = balance_scale.Class 

# Split dataset into training set and test set -> 70% training and 30% test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Pipeline Estimator

pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))

# Instantiate the bagging classifier

bgclassifier = BaggingClassifier(base_estimator=pipeline, n_estimators=100,max_features=4,max_samples=100,random_state=1, n_jobs=5)

# Fit the bagging classifier

bgclassifier.fit(X_train, y_train)

# Model scores on test data

test = bgclassifier.score(X_test, y_test)

# Predict the response for test dataset

y_pred = bgclassifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?

accuracy = metrics.accuracy_score(y_test, y_pred)

# Tase error

error = 1 - accuracy

# Show the result's classifier

print('##########\nBALANCE-SCALE DATA#####')
print('Test -> ', test); print('Error-> ', error)

### LOAD THE BREAST CANCER DATASET ####

bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Pipeline Estimator
pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=1))

# Fit the bagging classifier
bgclassifier.fit(X_train, y_train)

# Model scores on test and training data
test = bgclassifier.score(X_test, y_test)

# Predict the response for test dataset
y_pred = bgclassifier.predict(X_test)

# Model Accuracy, how often is the classifier correct?
accuracy = metrics.accuracy_score(y_test, y_pred)

# Tase error
error = 1 - accuracy

# print('Tamano dataset-> ', x.size); print('Test -> ', test); print('Train-> ', train);print('Error-> ', error)


