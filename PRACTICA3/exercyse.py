#!  /bin/python
# Author: Jaime Lorenzo Sanchez
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import metrics 
import matplotlib.pyplot as plt
from matplotlib import gridspec

# READ THE CREDIT CARD CSV FILE #
print('#### Credit card ####')
data = pd.read_csv('creditcard.csv')
print(pd.value_counts(data['Class'], sort = True))
# distribution of anomalous features
features = data.iloc[:,0:28].columns
# Determine number of fraud cases in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(Fraud)))
print('Valid Transactions: {}'.format(len(Valid)))

# READ THE balance_scale

print('#### balance_scale data #### ')
cols_names2 = ['Class','Left-Weight', 'Left-Distance','Right-Weight', 'Right-Distance']
balance_scale = pd.read_csv('balance-scale.data', header = None, names = cols_names2, sep = ",")
# Transform into binary classification
balance_scale['Class'] = [1 if b=='B' else 0 for b in balance_scale.Class]
print(balance_scale['Class'].value_counts())
Fraud2 = balance_scale[balance_scale['Class'] == 0] # Clase mayoritaria
Valid2 = balance_scale[balance_scale['Class'] == 1] # Clase minoritaria
outlier_fraction2 = len(Fraud2)/float(len(Valid2))
print(outlier_fraction2)
print('Fraud Cases: {}'.format(len(Fraud2)))
print('Valid Transactions: {}'.format(len(Valid2)))


# READ THE ABALONE.DATA

print('####ABALONE DATA ######')
cols_names3 = ['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

# Sex		nominal			M (mayoritaria), F (minoritaria) and I (infant)

abalone = pd.read_csv('abalone.data', header=None, names = cols_names3, sep=",")

# Transform into binary classification -> 1: Clase minoritario; 0 clase mayoritaria
abalone['Sex'] = [1 if f == 'F' else 0 for f in abalone.Sex]
print(abalone['Sex'].value_counts())
Fraud3 = abalone[abalone['Sex'] == 0] # Clase mayoritaria
Valid3 = abalone[abalone['Sex'] == 1] # Clase minoritaria
outlier_fraction3 = len(Fraud3)/float(len(Valid3))
print(outlier_fraction3)
print('Fraud Cases: {}'.format(len(Fraud3)))
print('Valid Transactions: {}'.format(len(Valid3)))

