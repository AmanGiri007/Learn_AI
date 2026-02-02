#accuracy: suitable for balanced datasets but misleadning for imbalanced data
#precision: important when false positives are costly (spam detection)
# recall(sensitivity): critical in situations where missing positives is costly
# f1score: useful for imbalanced datasets
# roc-auc: important for evaluating binary classifiers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#load dataset
data=load_iris()
X=data.data
y=(data.target==0).astype(int)

#split dataset
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

#train logistic regression model
model= LogisticRegression()
model.fit(X_train,y_train)

#predict
y_pred= model.predict(X_test)

#confusion matrix
cm= confusion_matrix(y_test,y_pred)
disp= ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Class 0","Class 0"])
disp.plot(cmap="Blues")
plt.title("Confustion Matrix")
plt.show()

print("\n Classification Report:")
print(classification_report(y_test,y_pred))