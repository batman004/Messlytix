#!/usr/bin/env python
# coding: utf-8

# In[5]:
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle


# In[6]:

data = pd.read_csv(r'DATASET.csv')
data.head()

# In[7]:


data['Menu Rating'].fillna(data['Menu Rating'].mean(), inplace=True)
data['Amount Of Food Cooked'].fillna(data['Amount Of Food Cooked'].mean(), inplace=True)
data['Wastage'].fillna(data['Wastage'].mean(), inplace=True)

# In[8]:
#Converting days of the week to numerical data

s=data['Day']
 
data.apply(lambda s: s.map({k:i for i,k in enumerate(s.unique())}))

# In[9]:


predict = "Amount Of Food Cooked"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[10]:


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.3)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)


# In[11]:

def refactoring(day):
    word_dict = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday': 5, 'Sunday':6}
    rating_dict = {0:7, 1:8.5, 2:9.1, 3:8.9, 4:8.6, 5: 7, 6:7.9}
    wastage_dict = {0:153.334, 1:143, 2:107.233, 3:102.223, 4:112.344, 5:349.456, 6:330.233}
    weekend = 1 if day in ['Saturday', 'Sunday'] else 0
    return list([word_dict[day],weekend,rating_dict[word_dict[day]],wastage_dict[word_dict[day]]])

# In[12]:


# DAY 0 :MONDAY ; DAY 1:TUESDAY ; DAY 2 :WEDNESDAY ; DAY 3: THURSDAY ; DAY 4:FRIDAY ; DAY 5:SATURDAY ; DAY 6 =SUNDAY
# IF IT IS A WEEKEND THEN TYPE 1, ELSE 0
# CHECK THE AMOUNT THE OF FOOD YOU NEED TO COOK TO MINIMISE YOUR WASTAGE
# By default wastage is 0 since we are trying to predict the ideal amounnt of food to be cooked


# In[13]:

# TESTING
Wastage = 0
# ------------------
input_list = refactoring(input('Enter the day of the week: '))
pred=linear.predict([input_list])
print('Menu rating for Today is :', input_list[2])
print('Average wastage on this day: ', input_list[3], 'Kgs')
print('To avoid this wastage,the predicted amount to be cooked :', pred, 'kgs')
pickle.dump(linear, open('messmodel.pkl', 'wb'))
