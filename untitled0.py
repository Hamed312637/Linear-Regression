# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:57:30 2023

@author: hamed
"""
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

#read data

path = 'D:\\labs\\Salary_Data.txt'
data = pd.read_csv(path, header=None, names=['experince','salary'])

# sperate data
cols = data.shape[1]
X=data.iloc[:,:cols-1]
y = data.iloc[:,cols-1:cols]

# linear regression

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=.2,random_state=143)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_prd = lr.predict(X_test)
print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)


# draw best fit line traning
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('For Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# draw best fit line testing
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, lr.predict(X_test), color = 'blue')
plt.title('For Testing Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# erorr
print("MSE :", mean_squared_error(y_test, y_prd))
print("R2 :", r2_score(y_test, y_prd))
