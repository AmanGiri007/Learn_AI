#accuracy, precision, recall, f1score
#cross validation: k-folds, stratified k-fold, leave one out cross validation
#confusion matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier

#load datasets
data=load_iris()
X,y= data.data,data.target

#initialize classifier
model= RandomForestClassifier(random_state=42)

#perform k-fold cross validation
kf= KFold(n_splits=5,shuffle=True,random_state=42)
cv_scores= cross_val_score(model,X,y,cv=kf,scoring="accuracy")

#output results
print("Cross-validation Scores:",cv_scores)
print("Mean Accuracy:",cv_scores.mean())