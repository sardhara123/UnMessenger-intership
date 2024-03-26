#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score


# In[2]:


df=pd.read_csv(r'C:\Users\rahul.sardhara\Desktop\Total Three Month Report\Rahul Data\Data Science\Car Dataset.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.isnull().sum().any()


# In[6]:


categorical_columns = df.select_dtypes(include=object).columns.tolist()
numerical_columns = df.select_dtypes(exclude=object).columns.tolist()
categorical_columns
numerical_columns


# In[7]:


label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
label_encoder


# In[8]:


scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])


# In[9]:


X = df.drop('selling_price', axis=1) 
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


model = RandomForestRegressor()


# In[11]:


model.fit(X_train, y_train)


# In[12]:


model.score(X_train, y_train)


# In[13]:


y_pred = model.predict(X_test)


# In[14]:


mse = mean_squared_error(y_test, y_pred)
r2_square = r2_score(y_test,y_pred)
print(f" R-squared: {r2_square}")
print(f'Mean Squared Error: {mse}')


# In[15]:


#Bayesian Regration#
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)


# In[17]:


model = BayesianRidge()
model


# In[18]:


model.fit(X_train, y_train)


# In[19]:


prediction = model.predict(X_test)


# In[20]:


print(f"Test Set r2 score : {r2_score(y_test, prediction)}")


# In[21]:


#Gradiant Boosting#

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as pt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



# In[22]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}


# In[23]:


reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))


# In[24]:


#Diffrance of above three Model Used#
Random Forest:

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

Gradient Boosting:
    
Gradient Boosting is another ensemble learning technique that builds trees sequentially, with each tree learning and correcting the errors made by the previous one

it's important to preprocess the data appropriately, including handling missing values, encoding categorical variables, and scaling numerical features. Additionally, cross-validation helps in estimating the model's performance more reliably and in selecting the best hyperparameters to avoid overfitting.

