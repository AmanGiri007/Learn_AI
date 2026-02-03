#Bagging(Bootstrap Aggregating); Random Forest
#Boosting: AdaBoost, Gradient Boosting, XGBoost
#Stacking: 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

data=load_iris()
X,y= data.data,data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#train individual models
log_model= LogisticRegression()
dt_model=DecisionTreeClassifier()
knn_model= KNeighborsClassifier()

log_model.fit(X_train,y_train)
dt_model.fit(X_train,y_train)
knn_model.fit(X_train,y_train)

#creating voting classifier
ensemble_model= VotingClassifier([
     ('log_reg',log_model),
     ('decision_tree',dt_model),
     ('knn',knn_model)
],
    voting='hard'
)

#train ensemble model
ensemble_model.fit(X_train,y_train)

y_pred_ensemble=ensemble_model.predict(X_test)

#evaluate accuracy
accuracy=accuracy_score(y_test,y_pred_ensemble)
print(f"Ensemble Model Accuracy:{accuracy:.2f}")

#evaluate individual models
y_pred_log=log_model.predict(X_test)
y_pred_dt=dt_model.predict(X_test)
y_pred_knn=knn_model.predict(X_test)

print(f"Logisitic regression accuracy:{accuracy_score(y_test,y_pred_log):.2f}")
print(f"Decision Tree accuracy:{accuracy_score(y_test,y_pred_dt):.2f}")
print(f"KNN Model accuracy:{accuracy_score(y_test,y_pred_knn):.2f}")
print(f"Ensemble Model accuracy:{accuracy:.2f}")