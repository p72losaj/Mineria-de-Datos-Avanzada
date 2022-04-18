#!  /bin/python
# Realizado por Jaime Lorenzo Sanchez

from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics


# Clasificador de regresion logistica
from sklearn.linear_model import LogisticRegression 


# Clasificador Arbol de decision
from sklearn.tree import DecisionTreeClassifier 

# Clasificador MLkNN
from skmultilearn.adapt import MLkNN


# Lista de clasificadores a utilizar
clasificadores = ['Logistic Regression', 'SVC', 'DecisionTreeClassifier','MLkNN']
# Lista de metricas
metricas = ['Hamming Loss', 'Accuracy','f1_micro']
# Lista de datos de la regresion logistica
lr = []

# lista de datos del clasificador Arbol de decision
tree = []
# Lista de datos del clasificador MLkNN
MLkNN = []

# Leemos el dataset birds
X_train, y_train, feature_names, label_names = load_dataset('birds', 'train')
X_test, y_test, _, _ = load_dataset('birds', 'test')
	
# Ejecutamos el metodo BR con Regresion Logistica
clf = LabelPowerset(classifier=LogisticRegression(max_iter=100000), require_dense=[False, True])
# Entrenamos el clasificador
clf.fit(X_train, y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas de la regresion logistica
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))

# Ejecutamos el metodo BR con el clasificador SVC
clf = LabelPowerset(classifier=SVC(), require_dense=[False, True])
# Entrenamos el clasificador
clf.fit(X_train, y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas del clasificador SVC
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))

# Ejecutamos el metodo BR del arbol de decision
clf = LabelPowerset(classifier = DecisionTreeClassifier())
# Entrenamos el clasificador
clf.fit(X_train,y_train)
# Calculamos la prediccion del test realizado
prediction = clf.predict(X_test)

# Calculamos las metricas del clasificador DecisionTreeClassifier
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))

# Ejecutamos el clasificador MlkNN

classifier = MLkNN(k=5)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)

# Calculamos las metricas del clasificador MLkNN
MLkNN.append(metrics.hamming_loss(y_test, prediction))
MLkNN.append(metrics.accuracy_score(y_test, prediction))
MLkNN.append(metrics.f1_score(y_test,prediction,average='micro'))

# Mostramos los datos obtenidos del dataset

for i in range (0,len(clasificadores)): 
	print('\t\t', clasificadores[i], end = " ")
print()

for j in range (0,len(metricas)):
	print(metricas[j], "\t", lr[j], "\t", svc[j], "\t", tree[j],MLkNN[j])	



