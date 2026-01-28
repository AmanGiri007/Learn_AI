import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

#generate synthetic dataset
np.random.seed(42)
n_samples=200
X=np.random.rand(n_samples,2)*10
y=(X[:,0]*1.5+X[:,1]>15).astype(int)

#create dataframe
df=pd.DataFrame(X,columns=['Age','Salary'])
df['Purchase']=y
X_train,X_test,y_train,y_test=train_test_split(df[['Age','Salary']],df['Purchase'],test_size=0.2,random_state=42)

#train logistic regression model
model=LogisticRegression()
model.fit(X_train,y_train)
#make predictions
y_pred=model.predict(X_test)

#evaluation performance
print('Accuracy:',accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))