#bagging : ensemble learning technique that trains multiple models
# on different subsets of the data
#reduces variance, improves robustness
#commonly used with decision trees, which are prone to high variance
#Random forest: builds multiple decision trees using bagging
# key parameters in Random Forests
# number of trees (n_estimators)
#maximum depth(max_depth)
#feature selection (max_features)
#minimum samples per leaf (min_samples_leaf)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data=load_breast_cancer()
X,y=data.data,data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Features",data.feature_names)
print("Classes:",data.target_names)

#train random forest
rf_model= RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)

y_pred= rf_model.predict(X_test)

#Evaluate performance
accuracy= accuracy_score(y_test,y_pred)
print('Random Forest Accuracy:',accuracy)
print("\n Classification report:\n",classification_report(y_test,y_pred))

#define hyperparameter grid
param_grid= {
    'n_estimators':(50,100,200),
    'max_depth':[None,10,20],
    'max_features':['sqrt','log2','None']
}

grid_search= GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train,y_train)

#display best parameters and score
print(f"Best Parameters:{grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy:{grid_search.best_score_}")