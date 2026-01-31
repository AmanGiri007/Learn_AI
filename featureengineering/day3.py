#encoding categorical variables
#commonly used for tree-based models, logisitc regression and neural networks
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import numpy as np

url="https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv"
df=pd.read_csv(url)

print("Dataset info:\n")
print(df.info())

print("\nDataset Preview\n")
print(df.head())

#apply onehot encoding
df_one_hot=pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)

print("\nOne-Hot Encoding\n")
print(df_one_hot.head())

#apply label encoding
label_encoder= LabelEncoder()
df['Pclass_encoded']=label_encoder.fit_transform(df['Pclass'])

#display encoded dataset
print("\n Label Encoded Dataset:")
print(df[['Pclass','Pclass_encoded']].head())

#apply frequency encoding 
df['Ticket_frequency']=df['Ticket'].map(df['Ticket'].value_counts())

#frequency encoded features
print("\n Frequency Encoded Feature:")
print(df[["Ticket","Ticket_frequency"]].head())

X=df_one_hot.drop(columns=['Survived','Name','Ticket','Cabin'])
y=df['Survived']

# X=X.apply(pd.to_numeric,errors='coerce')
# X.fillna(0)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
print("\nNaN Values for each columns",np.isnan(X_train).sum())
imputer= SimpleImputer(strategy='median')
X_train=imputer.fit_transform(X_train)
X_test= imputer.transform(X_test)


model= LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print("Accuracy Score:\n",accuracy_score(y_test,y_pred))