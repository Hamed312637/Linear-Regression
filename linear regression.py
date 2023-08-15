# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

  
# The data are given as list of lists (2d list)
path = 'D:\\labs\\Salary_Data.txt'
data = pd.read_csv(path, header=None, names=['experince','salary'])
#data.plot(kind='scatter', x='experince', y='salary', figsize=(5,5))
data.insert(0,"ones",1)
cols = data.shape[1]
X=data.iloc[:,:cols-1]
y = data.iloc[:,cols-1:cols]

#print(y)
X =np.matrix(X)
y = np.matrix(y)
theta = np.matrix(np.zeros(X.shape[1]))
def computecost(X,y,theta):
    z =np.power(((X*theta.T)-y),2)
    cost = np.sum(z)/(2*len(X))
    return cost
print(computecost(X, y, theta))
def gradientdecent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    paramters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X* theta.T)-y
        for j in range(paramters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X))*np.sum(term))
            theta = temp
            cost[i] = computecost(X, y, theta)
    return theta,cost
alpha = 0.01
iters = 10000
g,cost = gradientdecent(X, y, theta, alpha, iters) 
#print('theta',g)
#print('costs', cost)
print('computecost',computecost(X, y, g))

x = np.linspace(data.experince.min(), data.experince.max(), 100)
f = g[0, 0] + (g[0, 1] * x)
fig,ax = plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label= 'experince')
ax.scatter(data.experince, data.salary, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('experince')
ax.set_ylabel('salary')
ax.set_title('years of experince vs. salary')

fig ,ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')






       