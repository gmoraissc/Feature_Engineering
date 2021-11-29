grouped_feature_ls = []
correlated_groups = []

for feature in corrmat.feature1.unique():
  if feature not in grouped_feature_ls:

    correlated_block = corrmat[corrmat.feature1 == feature]
    grouped_feature_ls = grouped_feature_ls + list(
        correlated_block.feature2.unique()) + [feature]
    correlated_groups.append(correlated_block)

group = correlated_groups[2]

features = list(group.feature2.unique())

rf = RandomForestClassifier(n_estimators=200,
                        random_state=39,
                        max_depth=4)

rf.fit(X_train[features].fillna(0), y_train)

importance = pd.concat(
    [pd.Series(features),
     pd.Series(rf.feature_importances_)], axis=1)
importance.columns = ['feature', 'importance']
importance.sort_values(by='importance', ascending=False)
    
  
eature	importance
3	var_251	0.521389
1	var_56	0.195985
2	var_291	0.194187
0	var_236	0.088439

# para o grupo selecionado de variaveis que se correlacionam, var_251
# apresenta a maior importância para explicar a variável objetivo

from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection
)

import pandas as pd
import numpy as np
df = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')

x = df.loc[:, df.columns != 'target']
y = df.loc[:, df.columns == 'target']

X_train, X_test, y_train, y_test = train_test_split(x,
                                                   y,
                                                   test_size=0.3,
                                                    random_state=0)

from sklearn.pipeline import Pipeline

pipe = Pipeline([
                 ('constant', DropConstantFeatures(tol=0.998)),
                 ('duplicated', DropDuplicateFeatures()),
                 ('correlation', SmartCorrelatedSelection(selection_method='variance'))
])

pipe.fit(X_train)

X_train = pipe.transform(X_train)
X_test = pipe.transform(X_test)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

scaler = StandardScaler().fit(X_train)

logit = LogisticRegression(random_state=44, max_iter=500)
logit.fit(X_train, y_train)
pred = logit.predict_proba(X_test)
roc_auc_score(y_test, pred[:, 1])
