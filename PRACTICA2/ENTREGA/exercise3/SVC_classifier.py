#!  /bin/python
# Author: Jaime Lorenzo Sanchez

from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from skmultilearn.problem_transform import LabelPowerset

# SVC classifier
from sklearn.svm import SVC

# Metrics list
metricas = ['Hamming Loss', 'Accuracy','f1_micro']

# SVC list
svc = []

# First the birds dataset
X_train, y_train, feature_names, label_names = load_dataset('birds', 'train')
X_test, y_test, _, _ = load_dataset('birds', 'test')

# Print birds dataset size

print("Birds dataset size: ", )
