# Realizado por Jaime Lorenzo Sanchez
# version 1.0

import matplotlib.pyplot as plt

# Number of elements's datasets
nElements = [322,391,1211]

# Clasificador de Regresion Logistica

lr_hammingLoss = [0.062,0.220,0.094]
lr_accuracyScene = [0.467,0.332,0.681]
lr_f1Micro = [0.358,0.672,0.729]

# Clasificador SVC 

svc_hammingLoss = [0.056,0.450,0.086]
svc_accuracyScene = [0.477,0.129,0.708]
svc_f1Micro = [0.099,0.270,0.753]

# Clasificador Arbol de Decision

tree_hammingLoss = [0.067,0.290,0.156]
tree_accuracyScene = [0.471,0.198,0.511]
tree_f1Micro = [0.315,0.556,0.566]

# MLkNN classifier

mlknn_hammingLoss = [,0.298]
mlknn_accuracyScene = [,0.149]
mlknn_f1Micro = [,0.508]
# Hamming Loss

plt.plot(nElements, lr_hammingLoss, label = "Logistic Regression_LP")
plt.plot(nElements, svc_hammingLoss, label = "SVC_LP")
plt.plot(nElements, tree_hammingLoss, label = "DecisionTreeClassifier_LP")
plt.xlabel('Number of elements of dataset');plt.ylabel('Hamming Loss_LP');
plt.title('hamming loss multi-label problems')
plt.legend()
plt.show()

# Accuracy scene

plt.plot(nElements, lr_accuracyScene, label = "Logistic Regression_LP")
plt.plot(nElements, svc_accuracyScene, label = "SVC_LP")
plt.plot(nElements, tree_accuracyScene, label = "DecisionTreeClassifier_LP")
plt.xlabel('Number of elements of dataset');plt.ylabel('Accuracy Scene');
plt.title('Accuracy Scene multi-label problems')
plt.legend()
plt.show()

# f1_micro

plt.plot(nElements, lr_f1Micro, label = "Logistic Regression_LP")
plt.plot(nElements, svc_f1Micro, label = "SVC_LP")
plt.plot(nElements, tree_f1Micro, label = "DecisionTreeClassifier_LP")
plt.xlabel('Number of elements of dataset');plt.ylabel('f1_Micro');
plt.title('F1 Micro multi-label problemas')
plt.legend()
plt.show()


