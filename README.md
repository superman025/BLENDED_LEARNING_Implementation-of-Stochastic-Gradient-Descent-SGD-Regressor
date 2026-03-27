# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries like pandas, numpy, matplotlib, and sklearn
2. Load the dataset CarPrice_Assignment.csv
3. Display dataset details using head() and info(
4. Remove unnecessary columns (CarName, car_ID)
5. Convert categorical data into numerical form using one-hot encoding
6. 6.Split the dataset into features (X) and target variable (price).
7. Apply StandardScaler to normalize both X and y values.
8. 8.Split the data into training (80%) and testing (20%) sets.
9. 9.Train the model using SGD Regressor and make predictions on test data.
10. Evaluate the model using MSE, R², MAE and visualize actual vs predicted values




## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: sri jai v
RegisterNumber: 25018437
*/import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

data=data.drop(['CarName', 'car_ID'], axis=1)
data=pd.get_dummies(data,drop_first=True)

x=data.drop('price', axis=1)
y=data['price']

scaler = StandardScaler()
x = scaler.fit_transform(x)
y = scaler.fit_transform(np.array(y).reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sgd_model=SGDRegressor(max_iter=1000, tol=1e-3)

sgd_model.fit(x_train, y_train)

y_pred=sgd_model.predict(x_test)

mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)
mae=mean_absolute_error(y_test, y_pred)

print("\nName: SRIJAI V ")
print("Reg No: 25018437")
print("Mean Squared Error(y_test, y_pred):",mse)
print("R Squared Error:",r2)
print("Mean Absolute Error:",mae)

print("\nModel Coefficients:")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
```

## Output:
<img width="1037" height="349" alt="Screenshot 2026-02-23 142224" src="https://github.com/user-attachments/assets/7ac943f2-c5fd-4d3a-bc4a-7c426af38470" />
<img width="1083" height="417" alt="Screenshot 2026-02-23 142246" src="https://github.com/user-attachments/assets/45e7b38e-43ae-4fd4-8524-130f9f07ffc8" />



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
