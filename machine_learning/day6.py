#KNN algorithm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
data=load_iris()
X,y=data.data,data.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

#scale features
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#experiment with different values of k
# for k in range(1,11):
#     knn=KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X_train,y_train)
#     y_pred=knn.predict(X_test)
#     accuracy=accuracy_score(y_test,y_pred)
#     print(f"k={k},Accuracy={accuracy:.2f}")

model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
y_pred_log=model.predict(X_test)
accuracy_log=accuracy_score(y_test,y_pred_log)
print("Logistic regression accuracy:",accuracy_log)
print("Classfication Report:",classification_report(y_test,y_pred_log))

best_k=5
knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train,y_train)
knn_y_pred= knn.predict(X_test)
accuracy_knn= accuracy_score(y_test,knn_y_pred)
print(f"KNN Accuracy: {best_k}",accuracy_knn)
print("Classification Report:",classification_report(y_test,knn_y_pred))
