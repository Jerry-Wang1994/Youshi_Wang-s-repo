# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:11:26 2023

@author: Jerry
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
from sklearn.model_selection import train_test_split

df=pd.read_csv('hw5_treasury yield curve data.csv',index_col=['Date'])
df.index=pd.to_datetime(df.index)

#correlation heatmaps
cols_1=['SVENF01','SVENF02','SVENF03','SVENF04','SVENF05','SVENF06','Adj_Close']
train_cols1,test_cols1=train_test_split(df[cols_1],test_size=0.15,random_state=42)
cm_1 = np.corrcoef(train_cols1.values.T)
hm_1 = heatmap(cm_1,row_names=cols_1,column_names=cols_1)
plt.title('correlation bettwen 1-6 and Adj_Close')
plt.show()

cols_2=['SVENF06','SVENF07','SVENF08','SVENF09','SVENF10','SVENF11','Adj_Close']
train_cols2,test_cols2=train_test_split(df[cols_2],test_size=0.15,random_state=42)
cm_2 = np.corrcoef(df[cols_2].values.T)
hm_2 = heatmap(cm_2,row_names=cols_2,column_names=cols_2)
plt.title('correlation bettwen 6-11 and Adj_Close')
plt.show()

#randomly select some columns to see if they are indeed highly correlated
sample=df.sample(n=6,axis=1)
sample_cols=sample.columns.tolist()
cols_3=sample_cols + ['Adj_Close']
train_cols3,test_cols3=train_test_split(df[cols_3],test_size=0.15,random_state=42)
cm_3 = np.corrcoef(df[cols_3].values.T)
hm_3 = heatmap(cm_3,row_names=cols_3,column_names=cols_3)
plt.title('correlation bettwen 6 random features and Adj_Close')
plt.show()
#it seems that most features are strongly correlated to the target.
#but it doesnt mean that each feature has strong effect on the target.

#%%
#specify features and target, set train_test_split, and standardize data
X=df.iloc[:, :30].values
y=df['Adj_Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scalerX=StandardScaler().fit(X_train)
scalery=StandardScaler().fit(y_train.reshape(-1, 1))
X_train=scalerX.transform(X_train)
y_train=scalery.transform(y_train.reshape(-1, 1))
y_train = y_train.ravel()
X_test=scalerX.transform(X_test)
y_test=scalery.transform(y_test.reshape(-1, 1))

#%% linear regression model with SGD optimization algorithm
#with RMSE and R^2
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
def train_and_evaluate(clf,X_train,y_train):
    clf.fit(X_train,y_train)
    print("Coefficient of determination on training set:",clf.score(X_train,y_train))
    y_pred = clf.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE on training set:", rmse)
    cv = KFold(n_splits=5, shuffle=True, random_state=33)
    scores=cross_val_score(clf,X_train,y_train,cv=cv)
    print("Average coefficient of determination using 5-fold crossvalidation:",np.mean(scores))
    y_pred = cross_val_predict(clf, X_train, y_train, cv=cv)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE using 5-fold cross-validation:", rmse)
    
    return np.mean(scores), rmse

from sklearn import linear_model

#under no penalty
clf_sgd=linear_model.SGDRegressor(loss='squared_error',penalty=None,random_state=42)
train_and_evaluate(clf_sgd,X_train,y_train)
print(clf_sgd.coef_)

#linear regression model with SGD algorithm under l2 penalty
clf_sgd1=linear_model.SGDRegressor(loss='squared_error',penalty='l2',random_state=42)
train_and_evaluate(clf_sgd1,X_train,y_train)
print(clf_sgd1.coef_)

#measure the performance of linear regression models on test dataset.
from sklearn import metrics
def measure_performance(X,y,clf,show_accuracy=True,
                        show_classification_report=True,
                        show_confusion_matrix=True,
                        show_r2_score=False,
                        show_rmse=False):
    y_pred=clf.predict(X)
    if show_accuracy:
        print('Accuracy:{0:.3f}'.format(
            metrics.accuracy_score(y,y_pred)
            ),'\n')
    if show_classification_report:
        print('Classification report')
        print(metrics.classification_matrix(y,y_pred),'\n')
    if show_confusion_matrix:
        print('Confusion matrix')
        print(metrics.confusion_matrix(y,y_pred),'\n')
    if show_r2_score:
        print('Coefficient of determination on testing set:',(
            metrics.r2_score(y,y_pred)
            ),'\n')
    if show_rmse:
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        print('RMSE on testing set:', rmse, '\n')
    return metrics.r2_score(y,y_pred), rmse

#measure testing set performance
measure_performance(X_test,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

measure_performance(X_test,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

#%% SVM regression

from sklearn import svm
#using linear kernel
clf_svr=svm.SVR(kernel='linear')
train_and_evaluate(clf_svr,X_train,y_train)

#using non-linear kernel - polynomial
clf_svr_poly=svm.SVR(kernel='poly')
train_and_evaluate(clf_svr_poly,X_train,y_train)

#using non-linear kernel - radial basis function
clf_svr_rbf=svm.SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf,X_train,y_train)

#measure testing set performance
measure_performance(X_test,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

measure_performance(X_test,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

measure_performance(X_test,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

#%% dimentionality reduction - principal component analysis to 3 features

#perform eigendecomposition
cov_matrix=np.cov(X_train.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_matrix)
print('Eigenvalues \n%s' % eigen_vals)

#calculates explained variance and cumulative explained variance
total=sum(eigen_vals)
variance_explained=np.array([(i/total) for i in sorted(eigen_vals,reverse=True)])
cum_variance_explained=np.cumsum(variance_explained)

#visualize
plt.bar(range(1,31), variance_explained, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1,31), cum_variance_explained, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#feature transformation using first 3 principal components
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis],
               eigen_pairs[2][1][:, np.newaxis]))

print('Matrix W:\n', w)
#transform both training and testing dataset
X_train_pca = X_train.dot(w)
X_test_pca=X_test.dot(w)

#visualize transformed features
from mpl_toolkits.mplot3d import Axes3D
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for l, c, m in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_pca[y_train==l, 0],
               X_train_pca[y_train==l, 1],
               X_train_pca[y_train==l, 2],
               c=c, label=l, marker=m, alpha=0.5)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#%% train and test both models on new dataset with 3 principal components
train_and_evaluate(clf_sgd,X_train_pca,y_train)
measure_performance(X_test_pca,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

train_and_evaluate(clf_sgd1,X_train_pca,y_train)
measure_performance(X_test_pca,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

train_and_evaluate(clf_svr,X_train_pca,y_train)
measure_performance(X_test_pca,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

train_and_evaluate(clf_svr_poly,X_train_pca,y_train)
measure_performance(X_test_pca,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

train_and_evaluate(clf_svr_rbf,X_train_pca,y_train)
measure_performance(X_test_pca,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)

#explained variance for new dataset
cov_matrix_pca=np.cov(X_train_pca.T)
eigen_vals_pca,eigen_vecs_pca=np.linalg.eig(cov_matrix_pca)
total_pca=sum(eigen_vals_pca)
variance_explained_pca=np.array([(i/total_pca) for i in sorted(eigen_vals_pca,reverse=True)])
cum_variance_explained_pca=np.cumsum(variance_explained_pca)

#%%create worksheet to store performance
R2_untrans_sgd_train=train_and_evaluate(clf_sgd,X_train,y_train)[0] #instant
RMSE_untrans_sgd_train=train_and_evaluate(clf_sgd,X_train,y_train)[1]
R2_untrans_sgd_test=measure_performance(X_test,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_untrans_sgd_test=measure_performance(X_test,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_untrans_sgd1_train=train_and_evaluate(clf_sgd1,X_train,y_train)[0] #instant
RMSE_untrans_sgd1_train=train_and_evaluate(clf_sgd1,X_train,y_train)[1]
R2_untrans_sgd1_test=measure_performance(X_test,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_untrans_sgd1_test=measure_performance(X_test,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_untrans_svm_train=train_and_evaluate(clf_svr,X_train,y_train)[0] #41.84s
RMSE_untrans_svm_train=train_and_evaluate(clf_svr,X_train,y_train)[1]
R2_untrans_svm_test=measure_performance(X_test,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_untrans_svm_test=measure_performance(X_test,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_untrans_svmpoly_train=train_and_evaluate(clf_svr_poly,X_train,y_train)[0] #16.12
RMSE_untrans_svmpoly_train=train_and_evaluate(clf_svr_poly,X_train,y_train)[1]
R2_untrans_svmpoly_test=measure_performance(X_test,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_untrans_svmpoly_test=measure_performance(X_test,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_untrans_svmrbf_train=train_and_evaluate(clf_svr_rbf,X_train,y_train)[0] #5.19s
RMSE_untrans_svmrbf_train=train_and_evaluate(clf_svr_rbf,X_train,y_train)[1]
R2_untrans_svmrbf_test=measure_performance(X_test,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_untrans_svmrbf_test=measure_performance(X_test,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_trans_sgd_train=train_and_evaluate(clf_sgd,X_train_pca,y_train)[0] #instant
RMSE_trans_sgd_train=train_and_evaluate(clf_sgd,X_train_pca,y_train)[1]
R2_trans_sgd_test=measure_performance(X_test_pca,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_trans_sgd_test=measure_performance(X_test_pca,y_test,clf_sgd,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_trans_sgd1_train=train_and_evaluate(clf_sgd1,X_train_pca,y_train)[0] #instant
RMSE_trans_sgd1_train=train_and_evaluate(clf_sgd1,X_train_pca,y_train)[1]
R2_trans_sgd1_test=measure_performance(X_test_pca,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_trans_sgd1_test=measure_performance(X_test_pca,y_test,clf_sgd1,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_trans_svm_train=train_and_evaluate(clf_svr,X_train_pca,y_train)[0]   #41.77s
RMSE_trans_svm_train=train_and_evaluate(clf_svr,X_train_pca,y_train)[1]
R2_trans_svm_test=measure_performance(X_test_pca,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_trans_svm_test=measure_performance(X_test_pca,y_test,clf_svr,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_trans_svmpoly_train=train_and_evaluate(clf_svr_poly,X_train_pca,y_train)[0] #16.08s
RMSE_trans_svmpoly_train=train_and_evaluate(clf_svr_poly,X_train_pca,y_train)[1]
R2_trans_svmpoly_test=measure_performance(X_test_pca,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_trans_svmpoly_test=measure_performance(X_test_pca,y_test,clf_svr_poly,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]

R2_trans_svmrbf_train=train_and_evaluate(clf_svr_rbf,X_train_pca,y_train)[0] #6.28s
RMSE_trans_svmrbf_train=train_and_evaluate(clf_svr_rbf,X_train_pca,y_train)[1]
R2_trans_svmrbf_test=measure_performance(X_test_pca,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[0]
RMSE_trans_svmrbf_test=measure_performance(X_test_pca,y_test,clf_svr_rbf,
                    show_accuracy=False,show_classification_report=False,
                    show_confusion_matrix=False,show_r2_score=True,
                    show_rmse=True)[1]



data = [['linear',R2_untrans_sgd_train,RMSE_untrans_sgd_train,
          R2_untrans_sgd_test,RMSE_untrans_sgd_test,'instant'],
        ['linear L2',R2_untrans_sgd1_train,RMSE_untrans_sgd1_train,
         R2_untrans_sgd1_test,RMSE_untrans_sgd1_test,'instant'],
        ['svm', R2_untrans_svm_train,RMSE_untrans_svm_train,
         R2_untrans_svm_test,RMSE_untrans_svm_test,'41.84s'],
        ['svm poly',R2_untrans_svmpoly_train,RMSE_untrans_svmpoly_train,
         R2_untrans_svmpoly_test,RMSE_untrans_svmpoly_test,'16.12s'],
        ['svm rbf',R2_untrans_svmrbf_train,RMSE_untrans_svmrbf_train,
         R2_untrans_svmrbf_test,RMSE_untrans_svmrbf_test,'5.19s'],
        ['linear',R2_trans_sgd_train,RMSE_trans_sgd_train,
         R2_trans_sgd_test,RMSE_trans_sgd_test,'instant'],
        ['linear L2',R2_trans_sgd1_train,RMSE_trans_sgd1_train,
         R2_trans_sgd1_test,RMSE_trans_sgd1_test,'instant'],
        ['svm',R2_trans_svm_train,RMSE_trans_svm_train,
         R2_trans_svm_test,RMSE_trans_svm_test,'41.77s'],
        ['svm poly',R2_trans_svmpoly_train,RMSE_trans_svmpoly_train,
         R2_trans_svmpoly_test,RMSE_trans_svmpoly_test,'16.06s'],
        ['svm rbf',R2_trans_svmrbf_train,RMSE_trans_svmrbf_train,
         R2_trans_svmrbf_test,RMSE_trans_svmrbf_test,'6.28s']]


index = ['untransformed', 'untransformed', 'untransformed','untransformed',
         'untransformed','transformed','transformed','transformed',
         'transformed','transformed']
columns = ['model', 'R^2-train','RMSE-train','R^2-test','RMSE-test','training time']

df1 = pd.DataFrame(data, index=index, columns=columns)

print(df1)
df1.to_csv('performance.csv')

#%%
print("My name is Youshi Wang")
print("My NetID is: youshiw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")











