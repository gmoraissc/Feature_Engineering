REGULARISATION

Consists in adding a penalty on the different parameters of the model to reduce the freedom of the model.

The model will be less likely to fit the noise of the training data and will improve the generalization abilities of the model.

For linear models, there are in general 3 types of regularisation:

L1 (LASSO)
L2 (RIDGE)
L1/L2 (ELASTIC NET)
L1 - LASSO

Lasso will shrink some parameters to zero, therefore allowing for feature elimination.

L2 - RIDGE

As the penalisation increases, the coefficients approach but do not equal to zero, hence no variable is excluded.

By fitting a linear or logistic regression with a Lasso regularisation, the different variable coefficients can be evaluated and removed if equal to zero.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(f'/content/drive/MyDrive/data science/Datasets/paribas_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

X = data.drop(labels=['target', 'ID'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', max_iter=10000, solver='liblinear'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

selected_feat = X_train.columns[(sel_.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coeffs srhank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))

total features: 112
selected features: 90
features with coeffs srhank to zero: 22
  
  X_train_selected = sel_.transform(X_train.fillna(0))
X_test_selected = sel_.transform(X_test.fillna(0))

data = pd.read_csv(f'/content/drive/MyDrive/data science/Datasets/paribas_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

X = data.drop(labels=['target', 'ID'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(Lasso(alpha=100)) #high penalty, force the algorithm to shrink coeffs to zero 
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

selected_feat = X_train.columns[(sel_.get_support())]

print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coeffs srhank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))

#if the penalty is too high and important features are removed, one will notice
#a drop in the algorithm performance, which means regularisation must be decreased
