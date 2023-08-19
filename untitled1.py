# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:08:42 2023

@author: hamed
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error




path = 'C:\\Users\\hamed\\Desktop\\New folder\\Real estate.txt'
data = pd.read_csv(path, header=None, names=['no','transaction',' houseage',
                                            'distance MRT station',
                                            'n convenience stores',
                                            'latitude','longitude',
                                            'Y price'])

data.drop(['no'], axis=1, inplace=True)

data = (data-data.mean())/data.std()
#print(data.describe().T)
#print(data.info())
#print(data.isnull().sum())



#a = 'Y price'
#for no, i in enumerate(data.columns):
 #   if i != a:
  #      sns.histplot(data = data, x = i)
   #     plt.show()

cols = data.shape[1]
X= data.iloc[:,:cols-1]
y = data.iloc [:,cols-1:cols]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
   
lr =LinearRegression()
dt = DecisionTreeRegressor()
gbr = GradientBoostingRegressor()
rfr = RandomForestRegressor()
#xgb = XGBRegressor()
model = [lr, dt, gbr,  rfr]
results = pd.DataFrame(y_test.values.tolist(), columns=['Actual'])
accuracy = []
r2 = []
mean_er = []
for m in model:
    #print('Running', m)
    m.fit(X_train,y_train)
    predict = m.predict(X_test)
    results[str(m)] = pd.DataFrame(predict.tolist())
    accuracy.append([str(m), r2_score(y_test, predict), mean_squared_error(y_test, predict)])
    r2.append([str(m) , r2_score(y_test, predict)])
    mean_er.append([str(m) , mean_squared_error(y_test, predict)])
    #print(m, 'completed')   
    
results.columns = ['Actual','LR','DT','GBR','RFR']
print(results.head())

colu = ['LR','DT','GBR','RFR']
acc = pd.DataFrame(accuracy).T
acc = acc[1:]
acc.columns = colu
acc.index = ['R2', 'MSE']
print(acc)    

sns.barplot(acc[:1])
plt.title('R2')
plt.show()
sns.barplot(acc[1:])
plt.title('MSE')
plt.show()

for i in acc.columns:

    plt.figure(figsize=(5,5))
    plt.scatter(x = results['Actual'], y = results[i], alpha = 0.40)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(i)
    plt.show()



      