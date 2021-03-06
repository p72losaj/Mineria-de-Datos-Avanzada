import matplotlib.pyplot as plt # Graphics

# Number of elements's datasets

nElements = [750,3125,17070]

# bagging's classifier

test_bagging = [0.956, 0.883, 0.959];error_bagging = [0.044, 0.117,0.041]


# plt.plot(nElements,test_bagging,label = 'bgclassifier_test'); plt.plot(nElements,error_bagging,label='bgclassifier_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result') ; plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 

# RandomForest's classifier

test_randomForest = [0.956,0.819, 0.947]; error_randomForest = [0.044,0.181, 0.053]

# plt.plot(nElements,test_randomForest,label = 'randomForest_test'); plt.plot(nElements,error_randomForest,label='randomForest_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 

# RandomForest and bagging graphic

# plt.plot(nElements,test_randomForest,label = 'randomForest_test'); plt.plot(nElements,error_randomForest,label='randomForest_error') ; plt.plot(nElements,test_bagging,label = 'bgclassifier_test'); plt.plot(nElements,error_bagging,label='bgclassifier_error') ; plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation RandomForest and bagging classifiers') ; plt.legend(); plt.show() 

# Tree Decision's classifier

test_treeDecision = [0.956,0.793,0.930]; error_treeDecision = [0.045,0.207, 0.070]


# plt.plot(nElements,test_treeDecision,label = 'treeDecision_test'); plt.plot(nElements,error_treeDecision,label='treeDecision_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 

# Naïve Bayes's classifier

test_NaïveBayes = [0.933,0.904,0.947]; error_NaïveBayes = [0.067,0.096,0.053]

# plt.plot(nElements,test_NaïveBayes,label = 'NaïveBayes_test'); plt.plot(nElements,error_NaïveBayes,label='NaïveBayes_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show()

# AdaBoost's graphic

test_AdaBoost = [0.956,0.904,0.977]; error_AdaBoost = [0.044,0.096,0.023]

#plt.plot(nElements,test_AdaBoost,label = 'AdaBoost_test'); plt.plot(nElements,error_AdaBoost,label='AdaBoost_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show()  

# Tree Decision, Naïve_Bayes and AdaBoost's graphic

#plt.plot(nElements,test_AdaBoost,label = 'AdaBoost_test'); plt.plot(nElements,error_AdaBoost,label='AdaBoost_error'); plt.plot(nElements,test_treeDecision,label = 'treeDecision_test'); plt.plot(nElements,error_treeDecision,label='treeDecision_error');plt.plot(nElements,test_NaïveBayes,label = 'NaïveBayes_test');plt.plot(nElements,error_NaïveBayes,label='NaïveBayes_error');plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show()
 
# Evolution Error

# plt.plot(nElements,error_bagging,label='bagginError'); plt.plot(nElements,error_randomForest,label='error_randomForest'); plt.plot(nElements,error_treeDecision,label = 'error_error_treeDecision');plt.plot(nElements,error_NaïveBayes,label='error_NaïveBayes');plt.plot(nElements,error_AdaBoost, label='error_AdaBoost') ; plt.xlabel('Number of Elements');plt.ylabel('Error rate');plt.title('Error evolution of clasiffiers');plt.legend();plt.show()

# Stacking graphics

test_stacking = [0.974,0.878,0.965]; error_stacking = [0.026,0.122,0.035]

plt.plot(nElements, test_bagging,label='begging_test'); plt.plot(nElements,error_bagging,label='error_bagging'); plt.plot(nElements, test_AdaBoost,label='test_AdaBoost'); plt.plot(nElements, error_AdaBoost,label='error_AdaBoost'); plt.plot(nElements, test_stacking, label='test_stacking'); plt.plot(nElements, error_stacking, label='error_stacking'); plt.xlabel('Number of Elements');plt.ylabel('Results of clasiffiers');plt.title('Stacking comparation');plt.legend();plt.show()




