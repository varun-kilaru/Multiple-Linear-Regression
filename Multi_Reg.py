import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')

#read data
housing_data = pd.read_csv('housing.csv')

#info about data
print('\nInfo about data :')
print('#########################################')
print(housing_data.head())
print('#########################################\n')
print(housing_data.shape)
print('#########################################\n')
print(housing_data.info())
print('#########################################\n')
print(housing_data.describe())
print('#########################################\n')

#visualizing data

#visualizing numeric data
sns.pairplot(x_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'], y_vars = 'price', data = housing_data)
plt.show()
sns.pairplot(housing_data)
plt.show()

#visualizing categorical data
plt.figure(figsize = (20, 12))
plt.title('Categorical Data')
plt.subplot(2, 3, 1)
sns.boxplot(x='mainroad', y='price', data=housing_data)
plt.subplot(2, 3, 2)
sns.boxplot(x='guestroom', y='price', data=housing_data)
plt.subplot(2, 3, 3)
sns.boxplot(x='basement', y='price', data=housing_data)
plt.subplot(2, 3, 4)
sns.boxplot(x='hotwaterheating', y='price', data=housing_data)
plt.subplot(2, 3, 5)
sns.boxplot(x='airconditioning', y='price', data=housing_data)
plt.subplot(2, 3, 6)
sns.boxplot(x='furnishingstatus', y='price', data=housing_data)
plt.show()

#encoding the categorical variables

#encoding binary categorical variables
b_varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
housing_data[b_varlist] = housing_data[b_varlist].apply(lambda x: x.map({'yes' : 1, 'no' : 0}))
print(housing_data.head())

#encoding categorical variable
cat_var = pd.get_dummies(housing_data['furnishingstatus'])
housing_data = pd.concat([housing_data, cat_var], axis=1)
housing_data = housing_data.drop(['furnishingstatus', 'furnished'], axis=1)
print(housing_data.head())

#splitting & scaling our data

#splitting
df_train, df_test = train_test_split(housing_data, train_size=0.7, random_state=100)

#scaling the train data
scaler = MinMaxScaler()
vars_scaled = ['area', 'bathrooms', 'bedrooms', 'stories', 'parking', 'price']
df_train[vars_scaled] = scaler.fit_transform(df_train[vars_scaled])
print(df_train.head())

#scaling the test data
''' As the unseen data is used as test data, we don't know the min & max of the test
data to scale using normalization. So, we use the min & max values of the train data
to scale the test data also. Hence, the method transform() is used. '''

df_test[vars_scaled] = scaler.transform(df_test[vars_scaled])
print(df_test.head())

#identifying the correlations b/w predictors
plt.figure(figsize=(16, 10))
plt.title('Correlation b/w Data Attributes')
sns.heatmap(df_train.corr(), annot=True)
plt.show()

''' we build a sample model to select the significant features using
Recursive Feature Selection(RFE) '''
#building a sample model

#manipulating data
y_train = df_train.pop('price')
X_train = df_train

#our sample model
lm = LinearRegression()
model = lm.fit(X_train, y_train)

#using Recursive Feature Selection(RFE) to select Features
rfe = RFE(model, 10)
rfe = rfe.fit(X_train, y_train)

#creating a predictors table to choose predictors
rfe_df = pd.DataFrame()
rfe_df['Features'] = X_train.columns
rfe_df['Select/Not'] = rfe.support_
rfe_df['Ranking'] = rfe.ranking_
print(rfe_df)

#RFE selected features or predictors
selected_features = X_train.columns[rfe.support_]
print(selected_features)

#changing our data such that selected features are present
X_train = X_train[selected_features]

''' Here the RFE selects the following predictors:
['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
'hotwaterheating', 'airconditioning', 'parking', 'prefarea'].
But by building a model and checking the Variance Inflation Factor(VIF)
& p-values of selected predictors,we came to know that "bedrooms"
have the high p-value. So, we need to eliminate "bedrooms" predictor
which is insigificant. '''
X_train = X_train.drop('bedrooms', axis=1)

#building our original model
#manipualting data to use statsmodels api
X_train_sm = sm.add_constant(X_train)
print(X_train_sm.head())

#our original model
lr = sm.OLS(y_train, X_train_sm)
lr_model = lr.fit()

print(lr_model.summary())

#Creating a Variance Inflation Factor(vif) table for the predictors
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i)for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 3)
vif = vif.sort_values(by='VIF', ascending=False)
print(vif)

#Residual Analysis
y_train_pred = lr_model.predict(X_train_sm)
residuals = y_train - y_train_pred
plt.title('Residuals Distribution')
sns.distplot(residuals)
plt.show()

#prediction

#manipulating the test data
y_test = df_test.pop('price')
X_test = df_test

X_test = X_test[selected_features]
X_test = X_test.drop('bedrooms', axis=1)
X_test_sm = sm.add_constant(X_test)

#prediction on test data
y_test_pred = lr_model.predict(X_test_sm)

#evaluation
print('\nEvaluation of model :')
print('#########################################')
print('\nr2 score on train data : ', r2_score(y_true=y_train, y_pred=y_train_pred))
print('\nr2 score on test data : ', r2_score(y_true=y_test, y_pred=y_test_pred))
print('#########################################')