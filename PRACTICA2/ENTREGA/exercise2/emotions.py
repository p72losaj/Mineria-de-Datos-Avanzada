#!  /bin/python
# Practica2 ejercicio1 - Dataset emotions
# Realizado por Jaime Lorenzo Sanchez

from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance

# Clasificador de regresion logistica
from sklearn.linear_model import LogisticRegression 

# Clasificador SVC
from sklearn.svm import SVC

# Clasificador Arbol de decision
from sklearn.tree import DecisionTreeClassifier 

# Lista de clasificadores a utilizar
clasificadores = ['Logistic Regression', 'SVC', 'DecisionTreeClassifier']
# Lista de metricas
metricas = ['Hamming Loss', 'Accuracy','f1_micro']
# Lista de datos de la regresion logistica
lr = []
# Lista de datos del clasificador SVC
svc = []
# lista de datos del clasificador Arbol de decision
tree = []
# Leemos el dataset emotions
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

# Calculamos el tamano del dataset (numero de instancias)
emotions = 0

for x in X_train:
	emotions = emotions + 1
	
# Ejecutamos el metodo BR con Regresion Logistica
clf = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Entrenamos el clasificador
clf.fit(X_train, y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas de la regresion logistica
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))

# Ejecutamos el metodo BR con el clasificador SVC
clf = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
# Entrenamos el clasificador
clf.fit(X_train, y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas del clasificador SVC
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))

# Ejecutamos el metodo BR del arbol de decision
clf = BinaryRelevance(classifier = DecisionTreeClassifier())
# Entrenamos el clasificador
clf.fit(X_train,y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas del clasificadro DecisionTreeClassifier
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))

# Mostramos los datos obtenidos del dataset
print('Tamano del dataset: ', emotions)

for i in range (0,len(clasificadores)): 
	print('\t\t', clasificadores[i], end = " ")
print()

for j in range (0,len(metricas)):
	print(metricas[j], "\t", lr[j], "\t", svc[j], "\t", tree[j])	



