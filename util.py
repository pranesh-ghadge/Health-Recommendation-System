import pandas as pd
import numpy as np
from math import *
import tensorflow as tf
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

transactions = pd.read_csv('Transaction_Data2.csv')
policies = pd.read_csv('Policy_Info.csv')


label_encoder = preprocessing.LabelEncoder()
transactions['Gender']= label_encoder.fit_transform(transactions['Gender'])
# transactions['Residence']= label_encoder.fit_transform(transactions['Residence'])

transactions['PolicyName']= transactions['PolicyName'].str.replace("Policy_", "").astype("int")

features = []
for i in range(1, len(transactions.columns) - 1):
    features.append(transactions.columns[i])

X = transactions.loc[:, features]
y = transactions.loc[:, ["PolicyName"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3, train_size = .75)

rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train.values.ravel())