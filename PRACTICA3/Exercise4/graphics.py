#!  /bin/python
# Author: Jaime Lorenzo Sanchez
import matplotlib.pyplot as plt
# Graphic's original -> Axis x: Desequilibrio Axis y: Error
Desequilibrio = [232,527,1563]
SVC_error = [0.2338,0.8564, 0.7360]
RandomForest_Error = [0.2338,1.0,0.7679]
plt.plot(Desequilibrio,SVC_error,label='SVC_error');plt.plot(Desequilibrio,RandomForest_Error,label='RandomForest_error'); plt.xlabel('Class imbalance size'); plt.ylabel('Error'); plt.title('Imbalance class');plt.legend(); plt.show()
# OverSambling
oversambling = [500,576,2870]
SVC_error_oversambling = [0.2767,0.8723,0.6384]
RandomForest_error_oversambling = [0.1667,0.9947,0.6308]
plt.plot(oversambling,SVC_error_oversambling,label = 'SVC_error'); plt.plot(oversambling,RandomForest_error_oversambling,label = 'RandomForest_error');plt.xlabel('Class size');plt.ylabel('Error'); plt.title('Oversambling Imbalance class');plt.legend(); plt.show()
# Undersambling
undersambling = [49,268,1307]
SVC_error_undersambling = [0.8723,0.2857,0.7192]
RandomForest_error_undersambling = [1.0,0.2857,0.7230]
plt.plot(undersambling,SVC_error_undersambling,label='SVC_error'); plt.plot(undersambling,RandomForest_error_undersambling,label='RandomForest_error');plt.xlabel('Class size');plt.ylabel('Error'); plt.title('Undersambling Imbalance class');plt.legend();plt.show()
