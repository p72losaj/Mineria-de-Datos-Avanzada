# Practica 1- Ejercicio 1
# Compare el rendimiento de al menos 2 métodos no basados en boosting, como la mezcla
# por votación, bagging o random forest
# Realizado por Jaime Lorenzo Sanchez

# Tratamiento de datos

import numpy as np
import pandas as pd

# ======================================================
# Preprocesado y modelado

from sklearn import datasets
from sklearn import metrics 
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance

############################## breast cancer sklearn ###########################################

# Load the breast cancer dataset
bc = datasets.load_breast_cancer()
x = bc.data
y = bc.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Model scores on test and training data

breast_cancer.append(x.size) # Add the breast_cancer's size

breast_cancer.append(bgclassifier.score(X_test, y_test))

breast_cancer.append(bgclassifier.score(X_train, y_train))

bagging.append(breast_cancer) # Add the balance_scale results


print('Bagging ->  ', bagging) # Print the bagging's array



##########################################################################################
###########################################################################################

random_forest = [] # Random forest's array

iris_array2 = [] # iris's array

# split dataset in features and target variable

feature_cols = ['sepal length','sepal width', 'petal length','petal width']

x = iris[feature_cols] # Features

y = iris.label # Target variable

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123) # 70% training and 30% test

# Creación del modelo
# ==============================================================================
modelo = RandomForestRegressor(
            n_estimators = 10,
            criterion    = 'mse',
            max_depth    = None,
            max_features = 'auto',
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )
         
# Entrenamiento del modelo

#modelo.fit(X_train, y_train)

#balance_scale_array2 = [] # balance-scale's array

#breast_cancer2 = [] # breast cancer dataset's array


