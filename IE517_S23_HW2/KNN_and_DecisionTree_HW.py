# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:23:36 2023

@author: Jerry
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Task_1: plot impurity indices for probability range [0,1] for visual comparison of each impurity criteria.

#define functions gini, entropy, and error with parameter p.
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)   #creates evenly ranged array 'x' with values ranging from 0 to 1 and stepsize of 0.01

#plots 4 lines to diplay the measures of impurity; with labels, line styles and color for each line.
    #computes entropy for each value of 'p' in 'x';
    #scales the entropy by 0.5 for easier visual comparison
    #comptes classification error for each value of 'p' in 'x'
    #plots gini line for each value 'p'
ent = [entropy(p) if p != 0 else None for p in x]   
sc_ent = [e*0.5 if e else None for e in ent]      #scaled entropy for easier visual comparison with other 2 impurity lines
err = [error(i) for i in x]                         
fig = plt.figure()
ax = plt.subplot(111)
for i,lab,ls,c, in zip([ent,sc_ent,gini(x),err],
                       ['Entropy','Entropy (scaled)',
                        'Gini impurity',
                        'Misclassification error'],
                       ['-','-','--','-.'],
                       ['black','lightgray',
                        'r','g','c']):
    line=ax.plot(x,i,label=lab,
                 linestyle=ls,lw=2,color=c)

#add legends to display labels of each line;
    #2 horizontal lines to the plot
    #sets limit to y-axis
    #add labels for x and y axis
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),
          ncol=5,fancybox=True,shadow=True)
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])      #may change 1.1 to other values 
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')

#Task_2: Construct a decision tree classifier to the Treasury Squeeze dataset

from mlxtend.plotting import plot_decision_regions

df=pd.read_csv("Treasury Squeeze raw score data.csv")
print(df)   #see if the csv file is correctly loaded

#select 2 features, target, and split the data set into training and testing set
#test set will be 30% selected randomly.
X=df.loc[:,['price_crossing','roll_start']] 
y=df['squeeze'].astype(int)                 
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=33)
tree_model=DecisionTreeClassifier(criterion='gini',
                                  max_depth=4,
                                  random_state=1)
#use fit model of decision tree classifier to train X and y training dataset;
#combine training and testing set and plots the decision boundaries of trained model on combined data.
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined,
                      y_combined,
                      clf=tree_model,
                      )
#add labels, legend, and layout to the plot
plt.xlabel('price crossing')
plt.ylabel('roll start')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from sklearn import tree
#visualize the tree model
tree.plot_tree(tree_model)
plt.gcf().set_dpi(500) #if drop this argument, the plot will be very blurry.
plt.show()

#use Graphviz to obtain an image file of the trained decision tree model.
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
dot_data = export_graphviz(tree_model,
                           filled=True,
                           rounded=True,
                           class_names=['Squeeze',
                                        'non_Squeeze'],
                           feature_names=['price_crossing',
                                          'roll_start'],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

#construct random forest model and trains with training data, the algorithm improvies model performance by reducing overfitting compared to individual trees.
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      clf=forest)
plt.xlabel('price crossing')
plt.ylabel('roll start')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#task 3: Construct a KNN classifier to the Treasury Squeeze dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import metrics

#standardize the features of both training,test, and combined data for better comparison
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

#create KNN classifier to fit training data, with k value of 5 and power parameter of 2 for Minkowski metric.
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)
#plot decision regions with combined training and test data, with labels and legend.
plot_decision_regions(X_combined_std, y_combined,
                      clf=knn)
plt.xlabel('price crossing')
plt.ylabel('roll start')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#test k value from 1 to 25 and observe accuracy.
k_range=range(1,26)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    #check scores in Variables Explorer to find the k value with maximum accuracy:
        #in this case: k value of 21 is the best with accuracy 0.57+
#plot accuracy against k values for graphical visualization
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.show()

#can also test larger k values to see if there are even better accuracies
k_range=range(1,200)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
        #in this case: k value of 113 is the best with accuracy 0.61+
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.show()

#it does seem like k value of 113 reached the accuracy peak, try even larger k value(till 500):    
k_range=range(1,500)
scores=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
        #in this case: k value of 113 is the best with accuracy 0.61+
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.show()

#we are right, accuracy stays flat after certain k value till 500 and therefore k input of 113 is so far the best.

print("My name is Youshi Wang")
print("My NetID is: youshiw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
