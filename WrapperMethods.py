!pip install feature_engine
!pip install mlxtend

STEP FORWARD FEATURE SELECTION ALGORITHMS

F1, F2, Fn > evaluates (ex. roc-auc), selects best performer (ex. F1).

It then proceeds on to evaluating F1 with all the other features, one at a time, and evaluates.

The same procedure is done over and over until a given criteria is met (ex. Max number of features, best model performance..etc).

The ideal optimal would be when one feature is added and the performance dos not increase as much as a pre-definde theshold.

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector

data = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Datasets/HousePrices_HalfMil.csv')

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection
)

x = data.loc[:, data.columns != 'Prices']
y = data.loc[:, data.columns == 'Prices']

X_train, X_test, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=123
)

pipe = Pipeline([
                 ('constant', DropConstantFeatures(tol=0.998)),
                 ('duplicated', DropDuplicateFeatures()),
                 ('correlation', SmartCorrelatedSelection(selection_method='variance'))
])

pipe.fit(X_train)

X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

scaler = StandardScaler().fit(X_train)

sfs1 = SFS(RandomForestClassifier(n_jobs=4),
           k_features=10,
           forward=True,
           floating=False,
           verbose=2,
           scoring='roc_auc',
           cv=3)

sfs1 = sfs1.fit(np.array(X_train.fillna(0)), y_train.values.ravel())

selected_feat = X_train.columns[list(sfs1.k_feature_idx_)]

def run_randomforests(X_train, X_test, y_train, y_test):
  rf = RandomForestClassifier(n_estimators=200,
                              random_state=39,
                              max_depth=4)
  rf.fit(X_train, y_train)
  pred = rf.predict_proba(X_train)
  print(roc_auc_score(y_train, pred[:,1]))
  pred = rf.predict_proba(X_test)
  print(roc_auc_score(y_test, pred[:,1]))
  
run_randomforests(X_train[selected_feat].fillna(0),
                  X_test[selected_feat].fillna(0),
                  y_train, y_test)

STEP BACKWARD FEATURE SELECTION ALGORITHMS
x = data.loc[:, data.columns != 'Prices']
y = data.loc[:, data.columns == 'Prices']

X_train, X_test, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=123
)

sfs1 = SFS(RandomForestClassifier(n_jobs=2),
           k_features=10,
           forward=False,
           floating=False,
           verbose=2,
           scoring='roc_auc', #if a regression problem, score with r2
           cv=3)

sfs1 = sfs1.fit(np.array(X_train.fillna(0)), y_train.values.ravel())

selected_feat = X_train.columns[list(sfs1.k_feature_idx_)]

def run_randomforests(X_train, X_test, y_train, y_test):
  rf = RandomForestClassifier(n_estimators=200,
                              random_state=39,
                              max_depth=4)
  rf.fit(X_train, y_train)
  pred = rf.predict_proba(X_train)
  print(roc_auc_score(y_train, pred[:,1]))
  pred = rf.predict_proba(X_test)
  print(roc_auc_score(y_test, pred[:,1]))

run_randomforests(X_train[selected_feat].fillna(0),
                X_test[selected_feat].fillna(0),
                y_train, y_test)

EXHAUSTIVE FEATURE SELECTION

Finds all possible combinations with sub-sets of features (F1 + F2 + F3; F1; F2; F3; F1 + F2; F1 + F3...)

Its often impractical and computational expensive!

User can set a minimum and maximum number of subsets [5 - 10].


x = data.loc[:, data.columns != 'Prices']
y = data.loc[:, data.columns == 'Prices']

X_train, X_test, y_train, y_teste = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=123
)

pipe = Pipeline([
                 ('constant', DropConstantFeatures(tol=0.998)),
                 ('duplicated', DropDuplicateFeatures()),
                 ('correlation', SmartCorrelatedSelection(selection_method='variance'))
])

pipe.fit(X_train)

X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

scaler = StandardScaler().fit(X_train)

efs1 = EFS(RandomForestClassifier(n_jobs=2, random_state=0),
           min_features=2,
           max_features=5,
           scoring='roc_auc', #if a regression problem, score with r2 and use RandomForestRegressor
           print_progress=True,
           cv=3)

efs1 = efs1.fit(np.array(X_train.fillna(0)), y_train.values.ravel())

def run_randomforests(X_train, X_test, y_train, y_test):
  rf = RandomForestClassifier(n_estimators=200,
                              random_state=39,
                              max_depth=4)
  rf.fit(X_train, y_train)
  pred = rf.predict_proba(X_train)
  print(roc_auc_score(y_train, pred[:,1]))
  pred = rf.predict_proba(X_test)
  print(roc_auc_score(y_test, pred[:,1]))

selected_feat = X_train.columns[list(efs1.efs1.best_idx_)]

run_randomforests(X_train[selected_feat].fillna(0),
                X_test[selected_feat].fillna(0),
                y_train, y_test)
Ideal: when performance does not decrease beyond a user-given threshold.
