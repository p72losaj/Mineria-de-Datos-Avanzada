#!  /bin/python

from skmultilearn.dataset import load_dataset
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV

# Load emotions dataset example
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

# Print info
#print(feature_names)
#print(label_names)

# Try Binary relevance method with Logistic Regression
clf = BinaryRelevance(classifier=LogisticRegression(max_iter=10000), require_dense=[False, True])
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

print("BR with Logistic regression")
print("Hamming loss: ", metrics.hamming_loss(y_test, prediction))
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Coverage: ", metrics.coverage_error(y_test.toarray(), prediction.toarray()))

# MLkNN with cross-validation of k
parameters = {'k': range(1,11)}
score = 'f1_micro'

clf = GridSearchCV(MLkNN(), parameters, scoring=score)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)

print("MLkNN")
print("Best k: ", clf.best_params_['k'])
print("Hamming loss: ", metrics.hamming_loss(y_test, prediction))
print("Accuracy: ", metrics.accuracy_score(y_test, prediction))
print("Coverage: ", metrics.coverage_error(y_test.toarray(), prediction.toarray()))
