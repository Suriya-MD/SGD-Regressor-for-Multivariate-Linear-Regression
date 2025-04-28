# Exp-04: SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Fetch the California housing dataset and convert it to a DataFrame.
2. Select input features and target variables, then split data into training and testing sets.
3. Scale both input features and target variables using `StandardScaler`.
4. Initialize and train a `MultiOutputRegressor` with an `SGDRegressor` estimator.
5. Predict on the test data and inverse transform the predictions.
6. Calculate the Mean Squared Error (MSE) and display the first five predictions.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house
and number of occupants in the house with SGD regressor.
Developed by:  SURIYA M
RegisterNumber: 212223110055
*/
print("Name : SURIYA M")
print("Register Number : 212223110055")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
##Load the dataset California Housing
dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()
## first three features as input
x=df.drop(columns=['AveOccup','HousingPrice'])
## aveoccup and housingprice as output
y=df[['AveOccup','HousingPrice']]
## split the data into training and test data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
## scale the features and target variable
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.transform(y_test)
#initialize the SGDRegressor
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
#we need to use MultiOutputRegressor to handle multiple output variables
multi_output_sgd=MultiOutputRegressor(sgd)
#train the model
multi_output_sgd.fit(x_train,y_train)
#predict on the test data
y_pred=multi_output_sgd.predict(x_test)
#inverse transforms the predictions to get them back to the original scale
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
#evaluate the model using mse
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)
print("\nPredictions:\n",y_pred[:5])
print()
print("Name:  SURIYA M")
print("Reg No: 212223110055")
```

## Output:

### Head of the Dataset
![image](https://github.com/user-attachments/assets/3a991192-83ed-43cc-aca6-f4e0966db5ab)


### Mean Squared Error and Predicted Values
![image](https://github.com/user-attachments/assets/513ab4d2-2935-433c-9026-49596f31b71c)

## Result:

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
