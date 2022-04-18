# Graphics of exercise 2
# Author: Jaime Lorenzo Sanchez
import matplotlib.pyplot as plt

### TRAIN METRICS ####
# Birds dataset train: 322
# Emotions dataset train: 391
# Scene dataset train: 1211
train = [322,391,1211]
# HammingLoss metrics
lr_hammingLoss_train = [0.061,0.217,0.104]
svc_hammingLoss_train = [0.053,0.325,0.084]
tree_hammingLoss_train = [0.063,0.283,0.159]
lr_hammingLoss_lp_train = [,]
svc_hammingLoss_lp_train = [,]
tree_hammingLoss_lp_train = [,]
# Accuracy metrics
lr_accuracy_train = [0.464,0.257,0.523]
svc_accuracy_train = [0.471, 0.015, 0.587]
tree_accuracy_train = [0.406, 0.139, 0.352]
lr_accuracy_lp_train = [,]
svc_accuracy_lp_train = [,]
tree_accuracy_lp_train = [,]
# F1 micro metrics train
lr_micro_train = [0.381, 0.631,0.682]
svc_micro_train = [0.018, 0.075, 0.725]
tree_micro_train = [0.410, 0.564, 0.574]
lr_micro_lp_train = [,]
svc_micro_lp_train = [,]
tree_micro__lp_train = [,]

## TESTS METRICS ###
# Emotions dataset test:  202
# Birds dataset test:  323
# Scene dataset test:  1196 
tests = [202,323,1196]
# Hamming Loss metrics
lr_hammingLoss_test = [0.217,0.061,0.104]
svc_hammingLoss_test = [0.325,0.053,0.084]
tree_hammingLoss_test = [0.283,0.063,0.159]
lr_hammingLoss_lp_test = []
svc_hammingLoss_lp_test = []
tree_hammingLoss_lp_test = []
# Accuracy metrics
lr_accuracy_test = [0.257,0.464,0.523]
svc_accuracy_test = [0.015,0.471,0.587]
tree_hammingLoss_test = [0.139,0.406,0.352]
lr_accuracy_lp_test = []
svc_accuracy_lp_test = []
tree_hammingLoss_lp_test = []
# F1 micro metrics
lr_micro_test = [0.631,0.381,0.682]
svc_micro_test = [0.075,0.019,0.725]
tree_micro_test = [0.564,0.410,0.574]
lr_micro_lp_test = []
svc_micro_lp_test = []
tree_micro_test = []

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


