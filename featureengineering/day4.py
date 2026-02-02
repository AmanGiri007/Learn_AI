#techniques for feature selection 
#filter methods : correlation | mutual information ; quick evaluation of features before training a model
#wrapper methods : forward selection | backward elimination ; useful when feature interactions are important but computationally expensive
#embedded methods : lasso regression | tree-based models; effective when training tree-based models or regularized regression

#correlation and mutual information to select important features from a dataset
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data=load_diabetes()
df= pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

#display dataset information
print(df.head())
print(df.info())

#calcuate correlation matrix
correlation_matrix= df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#select features with high correlation to the target
correlated_features= correlation_matrix['target'].sort_values(ascending=False)
print("Features Most Correlated with Target:")
print(correlated_features)

#separate featured target
X=df.drop(columns=['target'])
y=df['target']

#calculate mutual information
mutual_info= mutual_info_regression(X,y)

#create a dataframe for visualization
mi_df=pd.DataFrame({'Feature':X.columns,"Mutual Information":mutual_info})
mi_df=mi_df.sort_values(by="Mutual Information",ascending=False)

print("Mutual Information Scores:")
print(mi_df)

#train a random forest model
model=RandomForestRegressor(random_state=42)
model.fit(X,y)

#get feature importance
feature_importance= model.feature_importances_
importance_df=pd.DataFrame({'Feature':X.columns,'Importance':feature_importance})
importance_df=importance_df.sort_values(by='Importance',ascending=False)

#plot feature 
plt.figure(figsize=(10,6))
plt.barh(importance_df['Features'],importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from Random Forest")
plt.show()