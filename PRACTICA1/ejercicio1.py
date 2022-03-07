# Practica 1
# Realizado por Jaime Lorenzo Sanchez

# Bibliotecas

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB # Import GaussianNB (a Naive Bayes classifier)
from sklearn import metrics #Import scikit­learn metrics module for accuracy calculation

# Leemos el archivo iris.data

cols_names = ['sepal length','sepal width', 'petal length','petal width', 'label']

archivo1 = pd.read_csv('iris.data', header = None, names = cols_names, sep = ",")

# split dataset in features and target variable

feature_cols = ['sepal length','sepal width', 'petal length','petal width']
	
X = archivo1[feature_cols] # Features
y = archivo1.label # Target variable

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)

# Predict the response for test dataset

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

nv = GaussianNB() # create a classifier

nv.fit(X_train,y_train) # fitting the data

y_pred = nv.predict(X_test) # store the prediction data


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Leemos el archivo balance-scale.data

cols_names = ['Class','Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']


archivo1 = pd.read_csv('balance-scale.data', header = None, names = cols_names, sep = ",")

# split dataset in features and target variable

feature_cols = ['Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']
	
X = archivo1[feature_cols] # Features
y = archivo1.Class # Target variable

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)

# Predict the response for test dataset

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

nv = GaussianNB() # create a classifier

nv.fit(X_train,y_train) # fitting the data

y_pred = nv.predict(X_test) # store the prediction data


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


