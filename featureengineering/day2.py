#minmax : knn or neural networks, sensitive to outliers
#z score: svm, logistic regression and PCA
# less sensitive to tree based models 

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler


#load dataset
data=load_iris()
X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

print("Dataset info:")
print(X.describe())
print("\n Target Classes:",data.target_names)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("Accuracy without scaling:",accuracy_score(y_test,y_pred))

#apply maxmix
scalar= MinMaxScaler()
X_scaled=scalar.fit_transform(X)

#split scaled data
X_train_scaled,X_test_scaled,y_train_scaled,y_test_scaled= train_test_split(X_scaled,y,test_size=0.2,random_state=42)

#train knn classified
knn_scaled=KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled,y_train_scaled)
y_pred_scaled=knn_scaled.predict(X_test_scaled)
print("Accuracy MinMax Scaling:",accuracy_score(y_test_scaled,y_pred_scaled))

#apply standard scalar
standard_scalar= StandardScaler()
X_standard= standard_scalar.fit_transform(X)

X_train_standard,X_test_standard,y_train_standard,y_test_standard= train_test_split(X_standard,y,test_size=0.2,random_state=42)

knn_standard= KNeighborsClassifier(n_neighbors=5)
knn_standard.fit(X_train_standard,y_train_standard)
y_pred_standard= knn_standard.predict(X_test_standard)
print("Accuracy Standard Scalar:",accuracy_score(y_test_standard,y_pred_standard))