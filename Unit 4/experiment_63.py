import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix,classification_report
actual=np.array([0,1,2,0,1,2,0,1,2,1,0,2])
pred=np.array([0,1,2,0,2,2,1,1,2,1,0,0])
print("Precision macro:",round(precision_score(actual,pred,average="macro"),3))
print("Recall macro:",round(recall_score(actual,pred,average="macro"),3))
print("F1 macro:",round(f1_score(actual,pred,average="macro"),3))
print("Confusion matrix:\n",confusion_matrix(actual,pred))
print(classification_report(actual,pred,target_names=["Billing","Delivery","Product issue"]))
