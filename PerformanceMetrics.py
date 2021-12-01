import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

data = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

x = data.loc[:, data.columns != 'target']
y = data.loc[:, data.columns == 'target']

X_train, X_test, y_train, y_test  = train_test_split(x,
                                                     y,
                                                     test_size=0.3,
                                                     random_state=32
)

roc_values = []

for feature in X_train.columns:
  clf = DecisionTreeClassifier()
  clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
  y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
  roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False).plot.bar(figsize=(20,8)) #roc_auc <0.5 are deciding at random

len(roc_values[roc_values <= 0.5]) / len(roc_values)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

x = data.loc[:, data.columns != 'target']
y = data.loc[:, data.columns == 'target']

X_train, X_test, y_train, y_test  = train_test_split(x,
                                                     y,
                                                     test_size=0.3,
                                                     random_state=32
)

mse_values = []

for feature in X_train.columns:
  clf = DecisionTreeRegressor()
  clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
  y_scored = clf.predict(X_test[feature].fillna(0).to_frame())
  mse_values.append(mean_squared_error(y_test, y_scored))

mse_values = pd.Series(mse_values)
mse_values.index = X_train.columns
mse_values.sort_values(ascending=False).plot.bar(figsize=(20,8)) #the smaller the mse, the better the model performance
