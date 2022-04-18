#!  /bin/python
# Author: Jaime Lorenzo Sanchez
from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from skmultilearn.adapt import MLkNN


# classifiers list
classifiers = ['Logistic Regression', 'SVC', 'DecisionTreeClassifier', 'MLKNN']
# metrics list
metric = ['Hamming Loss', 'Accuracy','f1_micro']

## DATASET EMOTIONS ##
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

# Method LabelPowerset with LogisticRegression
classifier = LabelPowerset(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr=[]
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with SVC classifier
classifier = LabelPowerset(classifier=SVC(), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# SVC metrics
svc=[]
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with DecisionTreeClassifier
classifier = LabelPowerset(classifier = DecisionTreeClassifier())
# Train and test
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
# DecisionTreeClassifier metrics
tree=[]
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))
# MLKNN classifier
classifier = MLkNN(k=3)
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# MLKNN metrics
mlknn = []
mlknn.append(metrics.hamming_loss(y_test, prediction))
mlknn.append(metrics.accuracy_score(y_test, prediction))
mlknn.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j], "\t", mlknn[j])	

## BIRDS DATASET ##
X_train, y_train, feature_names, label_names = load_dataset('birds', 'train')
X_test, y_test, _, _ = load_dataset('birds', 'test')

# Method LabelPowerset with LogisticRegression
classifier = LabelPowerset(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr=[]
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with SVC classifier
classifier = LabelPowerset(classifier=SVC(), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# SVC metrics
svc=[]
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with DecisionTreeClassifier
classifier = LabelPowerset(classifier = DecisionTreeClassifier())
# Train and test
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
# DecisionTreeClassifier metrics
tree=[]
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))
# MLKNN classifier
classifier = MLkNN(k=3)
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# MLKNN metrics
mlknn = []
mlknn.append(metrics.hamming_loss(y_test, prediction))
mlknn.append(metrics.accuracy_score(y_test, prediction))
mlknn.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j], "\t", mlknn[j])	
	
## DATASET SCENE ##
X_train, y_train, feature_names, label_names = load_dataset('scene', 'train')
X_test, y_test, _, _ = load_dataset('scene', 'test')

# Method LabelPowerset with LogisticRegression
classifier = LabelPowerset(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# Logistic Regression metrics
lr=[]
lr.append(metrics.hamming_loss(y_test, prediction))
lr.append(metrics.accuracy_score(y_test, prediction))
lr.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with SVC classifier
classifier = LabelPowerset(classifier=SVC(), require_dense=[False, True])
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# SVC metrics
svc=[]
svc.append(metrics.hamming_loss(y_test, prediction))
svc.append(metrics.accuracy_score(y_test, prediction))
svc.append(metrics.f1_score(y_test,prediction,average='micro'))
# Method LabelPowerset with DecisionTreeClassifier
classifier = LabelPowerset(classifier = DecisionTreeClassifier())
# Train and test
classifier.fit(X_train,y_train)
prediction = classifier.predict(X_test)
# DecisionTreeClassifier metrics
tree=[]
tree.append(metrics.hamming_loss(y_test, prediction))
tree.append(metrics.accuracy_score(y_test, prediction))
tree.append(metrics.f1_score(y_test,prediction,average='micro'))
# MLKNN classifier
classifier = MLkNN(k=3)
# Train and test
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
# MLKNN metrics
mlknn = []
mlknn.append(metrics.hamming_loss(y_test, prediction))
mlknn.append(metrics.accuracy_score(y_test, prediction))
mlknn.append(metrics.f1_score(y_test,prediction,average='micro'))

# Print the results
for i in range (0,len(classifiers)): 
	print('\t\t', classifiers[i], end = " ")
print()

for j in range (0,len(metric)):
	print(metric[j], "\t", lr[j], "\t", svc[j], "\t", tree[j], "\t", mlknn[j])	
	

print("END")
