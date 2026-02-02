#cross validation is used to asses how well a machine learning model generalizes to an independent dataset
#K-fold cross validation
#stratified K-fold
#leave one out cross validation
#hyperparameter tuning: set before training , tuning is crucial for optimizing model performance
#Grid search, random search

import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#loading titanic 
url='https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/titanic.csv'

df=pd.read_csv(url)
print(df.head())

#select relevant features
df=df[['Pclass','Sex','Age','Fare','Embarked','Survived']]

#handle missing values
df.fillna({'Age':df['Age'].median()},inplace=True)
df.fillna({'Embarked':df['Embarked'].mode(0)},inplace=True)

#define features and target
X=df.drop(columns=['Survived'])
y=df['Survived']

#apply feature scaling and encoding
preprocessor= ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),['Age','Fare']),
        ('cat',OneHotEncoder(),['Pclass','Sex','Embarked'])
    ]
)
X_preprocessed= preprocessor.fit_transform(X)

#train and evaluate logistic regression
log_model=LogisticRegression()
log_scores=cross_val_score(log_model,X_preprocessed,y,cv=5,scoring='accuracy')
print(f"Logistic Regression Accuracy:{log_scores.mean():.2f}")

#train and evaluate random forest
rf_model=RandomForestClassifier(random_state=42)
rf_scores= cross_val_score(rf_model,X_preprocessed,y,cv=5,scoring='accuracy')
print(f"Random Forest Accuracy:{rf_scores.mean():.2f}")


#define hyperparameter grid
param_grid={
    'n_estimators':[50,100,200],
    'max_depth':[None,10,20],
    'min_samples_split':[2,5,10]
}

#perform grid search
grid_search=GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=1
)

grid_search.fit(X_preprocessed,y)

#display best hyperparameter score
print(f"Best hyperparameters:{grid_search.best_params_}")
print(f"Best Accuracy:{grid_search.best_score_:.2f}")