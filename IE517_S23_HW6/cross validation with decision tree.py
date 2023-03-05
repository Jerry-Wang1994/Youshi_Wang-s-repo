# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 01:23:09 2023

@author: Jerry
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('ccdefault.csv',index_col=['ID'])
x=df.iloc[:,:23]
y=df['DEFAULT']

#%%run decision tree model and evaluate performance
score_table1 = pd.DataFrame(columns=['Random State', 'Train Score', 'Test Score'])
score_train_list = []
score_test_list = []
for random_state in range (1,11):
    X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.1, random_state=random_state)
    tree_model=DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=random_state)
    tree_model.fit(X_train,y_train)
    score_train=tree_model.score(X_train,y_train)
    score_test=tree_model.score(X_test,y_test)
    score_train_list.append(score_train)
    score_test_list.append(score_test)
    score_table1=pd.concat([score_table1,pd.DataFrame({'Random State': [random_state],'Train Score':[score_train], 'Test Score': [score_test]})], ignore_index=True)
    print(f"random_state = {random_state}, score_train={score_train},score_test={score_test}")
mean_train_score = np.mean(score_train_list)
std_train_score = np.std(score_train_list)
mean_test_score = np.mean(score_test_list)
std_test_score = np.std(score_test_list)
print(f"Mean score_train: {np.mean(score_train_list)}, Std score_train: {np.std(score_train_list)}")
print(f"Mean score_test: {np.mean(score_test_list)}, Std score_test: {np.std(score_test_list)}")
score_table1 = pd.concat([score_table1, pd.DataFrame({'Random State': ['Mean', 'Std'], 'Train Score': [mean_train_score, std_train_score], 'Test Score': [mean_test_score, std_test_score]})], ignore_index=True)
print(score_table1)

#%%run decision tree model with stratified k-fold cross-validation
score_table2 = pd.DataFrame(columns=['Random State', 'Train Score', 'Test Score'])
score_train_list = []
score_test_list = []
from sklearn.model_selection import StratifiedKFold, cross_val_score
for random_state in range (1,11):
    X_train,X_test,y_train,y_test=train_test_split(x, y, test_size=0.1, random_state=random_state)
    tree_model=DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=random_state)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state) #use stratified K-fold to handle imbalanced classes
    score_train = cross_val_score(tree_model, X_train, y_train, cv=cv).mean()
    score_test = cross_val_score(tree_model, X_test, y_test, cv=cv).mean()
    score_train_list.append(score_train)
    score_test_list.append(score_test)
    score_table2 = pd.concat([score_table2,pd.DataFrame({'Random State':[random_state],'Train Score':[score_train],'Test Score':[score_test]})],ignore_index=True)
    print(f"random_state = {random_state}, score_train={score_train},score_test={score_test}")
mean_train_score = np.mean(score_train_list)
std_train_score = np.std(score_train_list)
mean_test_score = np.mean(score_test_list)
std_test_score = np.std(score_test_list)
print(f"Mean score_train: {np.mean(score_train_list)}, Std score_train: {np.std(score_train_list)}")
print(f"Mean score_test: {np.mean(score_test_list)}, Std score_test: {np.std(score_test_list)}")
score_table2 = pd.concat([score_table2,pd.DataFrame({'Random State': ['Mean','std'],'Train Score': [mean_train_score,std_train_score],'Test Score':[mean_test_score,std_test_score]})],ignore_index=True)
print(score_table2)

print("My name is Youshi Wang")
print("My NetID is: youshiw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
