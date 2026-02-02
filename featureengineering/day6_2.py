from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

#load datasets
data=fetch_california_housing()
X,y=data.data,data.target 

#split dataset
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

#evaluate regression metrics
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)

print(f'Mean Absolute Error(MAE): {mae:.2f}')
print(f'Mean Squared Error(MSE):{mse:.2f}')
print(f'R2 Score: {r2:.2f}')
