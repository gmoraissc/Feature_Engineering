MUTUAL INFORMATION

How much information is gained when observing one variable in relation to another?

import pandas as pd
import numpy as np
df = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

x = df.loc[:, df.columns != 'target']
y = df.loc[:, df.columns == 'target']
X_train, X_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   test_size=0.3)

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

mi = mutual_info_classif(X_train.fillna(0), y_train)
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

from sklearn.feature_selection import SelectKBest, SelectPercentile

sel_ = SelectKBest(mutual_info_classif, k=10).fit(X_train.fillna(0), y_train)
X_train.columns[sel_.get_support()]
mi = mutual_info_regression(X_train.fillna(0), y_train)
mi = pd.Series(mi)
mi.index = X_train.columns
mi.sort_values(ascending=False)

mi.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ = SelectPercentile(mutual_info_regression, percentile=10).fit(X_train.fillna(0), y_train)
X_train.columns[sel_.get_support()]
