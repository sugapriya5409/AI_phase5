# AI_phase5


Data Source


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
     

data=pd.read_csv("USA_Housing.csv")

     

#understanding the data
data.info()
     
Data pre processing


data.head()
     

data.describe()
     
Feature selection


USA_Housing=pd.read_csv('/content/USA_Housing.csv')
sns.pairplot(USA_Housing)
     

sns.distplot(USA_Housing['Price'],hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')
     

#Displaying correlation among all the columns
USA_Housing.corr()
     

sns.heatmap(USA_Housing.corr(), annot = True)
     
training a linear regression model


#getting all column names
USA_Housing.columns



     

# Columns as Features
X = USA_Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
     

# Price is my Target Variable, what we trying to predict
y = USA_Housing['Price']
     
Model Training


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
     

#importing the Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
     

#creating LinearRegression Object
lm = LinearRegression()
     

#Training the Data Model
lm.fit(X_train, y_train)
     
Evaluation


#Displaying the Intercept
print(lm.intercept_)
     

coeff_data = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_data
     

#predictions
predictions = lm.predict(X_test)
     

plt.scatter(y_test, predictions, edgecolor='black')
     

sns.distplot((y_test - predictions), bins = 50, hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')
     

from sklearn import metrics
     

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
     
