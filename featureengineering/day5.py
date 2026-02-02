#feature creation: deriving new, meaningful features from existing ones to enhance
# a model's ability to capture important patterns in data
# adds domain knowledge to the dataset

# feature transformation: modifies existing features to better suit the learning algorithm
# logarithmic transformation, square root transformation, polynomial transformation
# enhances the models's ability to fit non-linear relationships
# transformations allow linear models to handle non-linear relationships
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df=pd.read_csv('bike_sharing_daily.csv')
print("Dataset Info:")
print(df.info())

#preview the first few rows
print("Dataset Preview:")
print(df.head())

#convert dteday to datetime
df['dteday']= pd.to_datetime(df['dteday'])

#create new features
df['day_of_week']= df['dteday'].dt.day_name()
df['month']=df['dteday'].dt.month
df['year']=df['dteday'].dt.year

#display the new features
print("\n New Features Derived from Date Column")
print(df[['dteday','day_of_week','month','year']].head())

#select feature and target
X=df[['temp']]
y=df['cnt']

#apply polynomial transformation
poly= PolynomialFeatures(degree=2,include_bias=False)
X_poly= poly.fit_transform(X)

#disply the transformed feature
print('\n Original and Polynomial Features')
print(pd.DataFrame(X_poly,columns=['temp','temp^2']).head())

#split the dataset
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
X_poly_train,X_poly_test= train_test_split(X_poly,test_size=0.2,random_state=42)

#train and evaluate model with original features
model_original= LinearRegression()
model_original.fit(X_train,y_train)
y_pred_original= model_original.predict(X_test)
mse_original= mean_squared_error(y_test,y_pred_original)

#train and evaluate model with polynomial featues
model_poly= LinearRegression()
model_poly.fit(X_poly_train,y_train)
y_pred_poly= model_poly.predict(X_poly_test)
mse_poly= mean_squared_error(y_test,y_pred_poly)

#compare results
print(f"MSE original:{mse_original:.2f}")
print(f"MSE Polynomial: {mse_poly:.2f}")