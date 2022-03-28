#!  /bin/python
# Practica2 ejercicio1 - Dataset delicious
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
# Leemos el dataset
X_train, y_train, feature_names, label_names = load_dataset('delicious', 'train')
X_test, y_test, _, _ = load_dataset('delicious', 'test')

# Calculamos el numero de instancias
delicious = 0
for i in X_train:
	delicious = delicious+1

# Ejecutamos el metodo BR con Regresion Logistica
clf = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
# Calculamos las metricas de la regresion logistica
lr2.append(metrics.hamming_loss(y_test, prediction))
lr2.append(metrics.accuracy_score(y_test, prediction))
lr2.append(metrics.f1_score(y_test,prediction,average='samples'))
# Mostramos los datos obtenidos del dataset delicious
print('Tamano del dataset: ',delicious)

for i in range (0,len(clasificadores)): 
	print('\t\t', clasificadores[i])
	for j in range (0,len(metricas)):
		print(metricas[j],"\t",lr[j])
