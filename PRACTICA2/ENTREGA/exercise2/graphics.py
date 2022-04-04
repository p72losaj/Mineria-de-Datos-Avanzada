# Realizado por Jaime Lorenzo Sanchez
# version 1.0

import matplotlib.pyplot as plt
# Number of elements's datasets
nElements = [322,391,1211]
# Clasificador de Regresion Logistica
lr_hammingLoss = [0.061,0.217, 0.104]; lr_accuracyScene = [0.464,0.257, 0.524]; lr_f1Micro = [0.381,0.631, 0.682]
# Clasificador SVC 
svc_hammingLoss = [0.053,0.325, 0.085]; svc_accuracyScene = [0.471,0.014, 0.524]; svc_f1Micro = [0.018,0.075, 0.725]
# Clasificador Arbol de Decision
tree_hammingLoss = [0.063,0.279, 0.156]; tree_accuracyScene = [0.384,0.134, 0.345]; tree_f1Micro = [0.423,0.562, 0.566]
# Hamming Loss exercise2
plt.plot(nElements, lr_hammingLoss, label = "Logistic Regression")
plt.plot(nElements, svc_hammingLoss, label = "SVC")
plt.plot(nElements, tree_hammingLoss, label = "DecisionTreeClassifier")
plt.xlabel('Number of elements of dataset');plt.ylabel('Hamming Loss');
plt.title('hamming loss multi-label problems')
plt.legend()
plt.show()
# Accuracy scene exercise 2
plt.plot(nElements, lr_accuracyScene, label = "Logistic Regression")
plt.plot(nElements, svc_accuracyScene, label = "SVC")
plt.plot(nElements, tree_accuracyScene, label = "DecisionTreeClassifier")
plt.xlabel('Number of elements of dataset');plt.ylabel('Accuracy Scene');
plt.title('Accuracy Scene multi-label problems')
plt.legend()
plt.show()

# f1_micro exercise2
plt.plot(nElements, lr_f1Micro, label = "Logistic Regression")
plt.plot(nElements, svc_f1Micro, label = "SVC")
plt.plot(nElements, tree_f1Micro, label = "DecisionTreeClassifier")
plt.xlabel('Number of elements of dataset');plt.ylabel('f1_Micro');
plt.title('F1 Micro multi-label problemas')
plt.legend()
plt.show()


