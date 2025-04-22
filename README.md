# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import pandas

 2.Import Decision tree classifier

 3.Fit the data in the model

 4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JASWANTH S
RegisterNumber:  212223220037

*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import numpy as np

data = pd.read_csv("/content/Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())

x = data[["Position", "Level"]]
y = data["Salary"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)
print(y_pred)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

new_data = pd.DataFrame([[5, 6]], columns=["Position", "Level"])
print("Prediction:", dt.predict(new_data))

```
## Output:

![image](https://github.com/user-attachments/assets/4c084163-f23c-4d5e-aad2-b8d3fc77ff19)

## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
