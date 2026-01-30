#supervised learning project
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

#load telco customer churn
df_telco=pd.read_csv('Telco-Customer-Churn.csv')
# df_telco.drop(columns=['customerID'])
# print(df_telco.isnull().sum())
# for i in df_telco.columns:
#     df_telco[i].fillna(max)
# df_telco.fillna(df_telco.mean(),inplace=True)
#encode categorical variables
le=LabelEncoder()
for col in df_telco.columns:
    df_telco[col]=le.fit_transform(df_telco[col])

X=df_telco.drop(columns=['Churn'])
X=X.apply(pd.to_numeric,errors='coerce') #invalid -> NaN
X=X.fillna(0)
y=df_telco['Churn']

scalar=StandardScaler()
X=scalar.fit_transform(X)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

log_model= LogisticRegression(max_iter=200)
log_model.fit(X_train,y_train)

knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,y_train)

#evaluate models
log_pred=log_model.predict(X_test)
knn_pred= knn_model.predict(X_test)

print("\nLogistic Regression Classification reports:")
print(classification_report(y_test,log_pred))

print("\n KNN Classification Model:")
print(classification_report(y_test,log_pred))

print('Confusion Matrix\n',confusion_matrix(y_test,knn_pred))
print(df_telco.info())
print(df_telco.describe())

#visualize
sns.countplot(x='Churn',data=df_telco)
plt.title("Churn distribution")
plt.show()


