import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Example test labels and predictions from a binary normal/abnormal classifier.
y_true = np.array([0,0,1,1,0,1,0,1,1,0])
y_pred = np.array([0,0,1,0,0,1,1,1,1,0])
print('Accuracy:', round(accuracy_score(y_true,y_pred),3))
print('Precision:', round(precision_score(y_true,y_pred),3))
print('Recall:', round(recall_score(y_true,y_pred),3))
print('F1-score:', round(f1_score(y_true,y_pred),3))
print('Confusion matrix:\n', confusion_matrix(y_true,y_pred))
print(classification_report(y_true,y_pred,target_names=['Normal','Abnormal']))
