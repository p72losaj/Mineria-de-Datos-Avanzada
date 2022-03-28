#!  /bin/python
# Practica2 ejercicio1 - Dataset medical
# Realizado por Jaime Lorenzo Sanchez

from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression

# Lista de clasificadores a utilizar
clasificadores = ['Logistic Regression']
# Lista de metricas
metricas = ['Hamming Loss', 'Accuracy','f1_score']
# Lista de datos de la regresion logistica
lr = []

# Leemos el dataset medical

X_train, y_train, feature_names, label_names = load_dataset('medical', 'train')
X_test, y_test, _, _ = load_dataset('medical', 'test')

# Calculamos el numero de instancias
medical = 0
for i in X_train:
	medical = medical+1

# Ejecutamos el metodo BR con Regresion Logistica
clf = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

