#!  /bin/python
# Author: Jaime Lorenzo Sanchez
from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
# classifiers list
classifiers = ['Logistic Regression', 'SVC', 'DecisionTreeClassifier']
lr=[]
svc = []
tree = []
# metrics list
metric = ['Hamming Loss', 'Accuracy','f1_micro']

## DATASET EMOTIONS ##
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')
# Number of train's dataset and test's dataset
trains = 0; tests = 0
for x in X_train:
	trains = trains + 1
for y in X_test:
	tests = tests+1
print("Emotions dataset train: ", trains);print("Emotions dataset test: ", tests)

# Method BR with LogisticReggression classifier
classifier = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with SVC classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
# train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# SVC metrics
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with DecisionTreeClassifier
classifier = BinaryRelevance(classifier = DecisionTreeClassifier())
# Train and test
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
# DecisionTreeClassifier metrics
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j])	


## BIRDS DATASET ##
X_train, y_train, feature_names, label_names = load_dataset('birds', 'train')
X_test, y_test, _, _ = load_dataset('birds', 'test')
# Number of train's dataset and test's dataset
trains = 0; tests = 0
for x in X_train:
	trains = trains + 1
for y in X_test:
	tests = tests+1
print("Birds dataset train: ", trains);print("Birds dataset test: ", tests)

# Method BR with LogisticReggression classifier
classifier = BinaryRelevance(classifier=LogisticRegression(max_iter=100000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr = []
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with SVC classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
# train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# SVC metrics
svc = []
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with DecisionTreeClassifier
classifier = BinaryRelevance(classifier = DecisionTreeClassifier())
# Entrenamos el clasificador
classifier.fit(X_train,y_train)
# Calculamos la prediccion del test realizado
prediction = classifier.predict(X_test)
# DecisionTreeClassifier metrics
tree = []
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j])	


## DATASET SCENE ##
X_train, y_train, feature_names, label_names = load_dataset('scene', 'train')
X_test, y_test, _, _ = load_dataset('scene', 'test')
# Number of train's dataset and test's dataset
trains = 0; tests = 0
for x in X_train:
	trains = trains + 1
for y in X_test:
	tests = tests+1
print("Scene dataset train: ", trains);print("Scene dataset test: ", tests)

# Method BR with LogisticRegression
classifier = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr=[]
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with SVC classifier
classifier = BinaryRelevance(classifier=SVC(), require_dense=[False, True])
# Entrenamos el clasificador
classifier.fit(X_train, y_train)
# Calculamos la prediccion del test realizado
prediction = classifier.predict(X_test)

# Classifier SVC metrics
svc = []
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))

# Method BR with DecisionTreeClassifier
classifier = BinaryRelevance(classifier = DecisionTreeClassifier())
# Entrenamos el clasificador
classifier.fit(X_train,y_train)
# Calculamos la prediccion del test realizado
prediction = classifier.predict(X_test)

# DecisionTreeClassifier metrics
tree = []
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j])	

print("END")
