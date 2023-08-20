import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import re


df_train = pd.read_csv(os.path.join('data','train.csv'))
df_test = pd.read_csv(os.path.join('data','test.csv'))

submission_df = pd.DataFrame(index=df_test['PassengerId'])

def get_title(name):
    title_list = ["Mr.", "Mrs.", "Miss.", "Master.", "Don.", "Rev.", "Dr.", "Mme.", "Ms.", 
        "Major.", "Lady.", "Sir.", "Mlle.", "Col.", "Capt.", "Countess.", "Jonkheer.", "Dona."]
    
    for title in title_list:
        if title in name:
            return title
    return None


def get_floor(cabin):
    """Returns a tuple of cabin floor and the number of the cabin"""
    if cabin == "null": return ["null",0]

    data = [re.findall("[a-zA-Z]+",cabin)[-1]]

    num = re.findall("\d+",cabin)
    if num:
        data.append(int(num[-1]))
    else:
        data.append(0)

    return data

def process(df):
    """This is going to do some feature engineering with the data to get better results"""
    ## Filling null values. Starting with Cabin feature, then age and embarked
    df["Cabin"].fillna("null", inplace=True)

    ## Filled age with 0 as the coefficient multiplying with 0 is going to have no effect, so as
    ## to make it so the age is not known, it won't impact the predictions (it might though)
    df["Age"].fillna(0,inplace=True)
    df["Embarked"].fillna("null",inplace=True)
    df["Fare"].fillna(0,inplace=True)

    ## Adding features to split up cabin into discrete and continuous features
    result_series = df["Cabin"].apply(lambda cabin: get_floor(cabin))
    df["Cabin Floor"] = result_series.str[0]
    df["Cabin Room"] = result_series.str[1]
    
    ## Going to add a new feature to represent the title of people, there might be an 
    ## important detail to predict survavibility
    df["Title"] = df["Name"].apply(lambda name: get_title(name))

    ## Dropping name afterwards as it is not going to help us
    df.drop("Cabin", axis=1, inplace=True)
    df.drop("Name", axis=1, inplace=True)
    df.drop(["PassengerId", "Ticket"], axis=1, inplace=True)

    
    return df

df_train = process(df_train)
df_test = process(df_test)

y_train = df_train["Survived"]
x_train = df_train.drop("Survived",axis=1)
x_train = pd.get_dummies(x_train, dtype=int)

x_test = pd.get_dummies(df_test, dtype=int)

## Fitting the same columns to one another
train_col = x_train.columns
test_col = x_test.columns

remove_feat = [col_name for col_name in train_col if col_name not in test_col]
x_train_fit = x_train.drop(remove_feat, axis=1)

col_to_add = [col for col in test_col if col not in train_col]
index = test_col.get_indexer(col_to_add)[0]

x_train_fit.insert(int(index),col_to_add[0],0)

## Need to scale the data

scaler = StandardScaler()

scaled_train = pd.DataFrame(scaler.fit_transform(x_train_fit), columns=x_train_fit.columns)
scaled_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)

## Getting result
    ## Logistic Regression
model = LogisticRegression()
model.fit(scaled_train,y_train)
result = model.predict(scaled_test)

    ## Random Forest
rf_model = RandomForestClassifier(100)
rf_model.fit(scaled_train,y_train)
rf_result = rf_model.predict(scaled_test)

## Saving submission
submission_df["Survived"] = result
submission_df.to_csv(os.path.join('results','result.csv'))

submission_df["Survived"] = rf_result
submission_df.to_csv(os.path.join('results','result_rf.csv'))
