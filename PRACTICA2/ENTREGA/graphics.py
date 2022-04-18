# Graphics of exercise 2
# Author: Jaime Lorenzo Sanchez
import matplotlib.pyplot as plt

### TRAIN METRICS ####
train = [322,391,1211]
# HammingLoss metrics
lr_hammingLoss_train = [0.061,0.217,0.104]
svc_hammingLoss_train = [0.053,0.325,0.084]
tree_hammingLoss_train = [0.063,0.283,0.159]

lr_hammingLoss_lp_train = [0.062,0.220,0.095]
svc_hammingLoss_lp_train = [0.056,0.450,0.086]
tree_hammingLoss_lp_train = [0.069,0.290, 0.152]
mlknn_hammingLoss_train = [0.062,0.295, 0.102]

# Accuracy metrics
lr_accuracy_train = [0.464,0.257,0.523]
svc_accuracy_train = [0.471, 0.015, 0.587]
tree_accuracy_train = [0.406, 0.139, 0.352]

lr_accuracy_lp_train = [0.464,0.332, 0.681]
svc_accuracy_lp_train = [0.477,0.129, 0.708]
tree_accuracy_lp_train = [0.461,0.208, 0.517]
mlknn_accuracy_train = [0.452,0.193, 0.558]
# F1 micro metrics train
lr_micro_train = [0.381, 0.631,0.682]
svc_micro_train = [0.018, 0.075, 0.725]
tree_micro_train = [0.410, 0.564, 0.574]

lr_micro_lp_train = [0.354,0.672, 0.729]
svc_micro_lp_train = [0.099,0.270,0.753]
tree_micro_lp_train = [0.294,0.559, 0.575]
mlknn_micro_train = [0.132,0.557, 0.685]
	 		
## TESTS METRICS ###

tests = [202,323,1196]
# Hamming Loss metrics
lr_hammingLoss_test = [0.217,0.061,0.104]
svc_hammingLoss_test = [0.325,0.053,0.084]
tree_hammingLoss_test = [0.283,0.063,0.159]

lr_hammingLoss_lp_test = [0.220,0.062,0.095]
svc_hammingLoss_lp_test = [0.450,0.056,0.086]
tree_hammingLoss_lp_test = [0.290,0.069, 0.152]
mlknn_hammingLoss_test = [0.295, 0.062, 0.102]

# Accuracy metrics
lr_accuracy_test = [0.257,0.464,0.523]
svc_accuracy_test = [0.015,0.471,0.587]
tree_hammingLoss_test = [0.139,0.406,0.352]

lr_accuracy_lp_test = [0.332,0.464, 0.681]
svc_accuracy_lp_test = [0.129,0.477, 0.708]
tree_hammingLoss_lp_test = [0.208,0.461, 0.517]
mlknn_hammingLoss_test = [0.193,0.452, 0.558]

# F1 micro metrics
lr_micro_test = [0.631,0.381,0.682]
svc_micro_test = [0.075,0.019,0.725]
tree_micro_test = [0.564,0.410,0.574]

lr_micro_lp_test = [0.672,0.354, 0.729]
svc_micro_lp_test = [0.270,0.099, 0.753]
tree_micro_test = [0.559,0.294, 0.575]
mlknn_micro_test = [0.516,0.132, 0.685]




# Hamming Loss exercise2
#plt.plot(nElements, lr_hammingLoss, label = "Logistic Regression")
#plt.plot(nElements, svc_hammingLoss, label = "SVC")
#plt.plot(nElements, tree_hammingLoss, label = "DecisionTreeClassifier")
#plt.xlabel('Number of elements of dataset');plt.ylabel('Hamming Loss');
#plt.title('hamming loss multi-label problems')
#plt.legend()
#plt.show()
# Accuracy scene exercise 2
#plt.plot(nElements, lr_accuracyScene, label = "Logistic Regression")
#plt.plot(nElements, svc_accuracyScene, label = "SVC")
#plt.plot(nElements, tree_accuracyScene, label = "DecisionTreeClassifier")
#plt.xlabel('Number of elements of dataset');plt.ylabel('Accuracy Scene');
#plt.title('Accuracy Scene multi-label problems')
#plt.legend()
#plt.show()

# f1_micro exercise2
#plt.plot(nElements, lr_f1Micro, label = "Logistic Regression")
#plt.plot(nElements, svc_f1Micro, label = "SVC")
#plt.plot(nElements, tree_f1Micro, label = "DecisionTreeClassifier")
#plt.xlabel('Number of elements of dataset');plt.ylabel('f1_Micro');
#plt.title('F1 Micro multi-label problemas')
#plt.legend()
#plt.show()


