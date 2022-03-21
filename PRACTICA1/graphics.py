import matplotlib.pyplot as plt # Graphics

# Number of elements's datasets

nElements = [750,3125,17070]

# bagging's classifier

test_bagging = [0.956, 0.883, 0.959];error_bagging = [0.044, 0.117,0.041]


# RandomForest's classifier

test_randomForest = [0.956,0.819, 0.947]; error_randomForest = [0.044,0.181, 0.053]

# Tree Decision's classifier

test_treeDecision = [0.956,0.793]; error_treeDecision = [0.045,0.207]


# Bagging's graphic

# plt.plot(nElements,test_bagging,label = 'bgclassifier_test'); plt.plot(nElements,error_bagging,label='bgclassifier_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result') ; plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 

# RandomForest's graphics

# plt.plot(nElements,test_randomForest,label = 'randomForest_test'); plt.plot(nElements,error_randomForest,label='randomForest_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 

# RandomForest and bagging graphic

plt.plot(nElements,test_randomForest,label = 'randomForest_test'); plt.plot(nElements,error_randomForest,label='randomForest_error') ; plt.plot(nElements,test_bagging,label = 'bgclassifier_test'); plt.plot(nElements,error_bagging,label='bgclassifier_error') ; plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation RandomForest and bagging classifiers') ; plt.legend(); plt.show() 

# Tree's graphic

#plt.plot(nElements,test_treeDecision,label = 'treeDecision_test'); plt.plot(nElements,error_treeDecision,label='treeDecision_error'); plt.xlabel('Number of elements of dataset'); plt.ylabel('Rate of the result'); plt.title('Relation Dataset and classifier result'); plt.legend(); plt.show() 


