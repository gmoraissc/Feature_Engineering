import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv(f'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
data = data.replace('?', np.nan)
data.dropna(subset=['embarked', 'fare', 'cabin'], inplace=True)
data['age'] = data['age'].astype('float')
data['age'] = data['age'].fillna(data['age'].mean())

Y = data.loc[:, data.columns == 'survived']

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    Y,
                                                    test_size=0.3,
                                                    random_state=3)

def mean_encoding(df_train, df_test, categorical_vars):

  df_train_temp = df_train.copy()
  df_test_temp = df_test.copy()

  for col in categorical_vars:

    target_mean_dict = df_train.groupby([col])['survived'].mean().to_dict()

    df_train_temp[col] = df_train[col].map(target_mean_dict)
    df_test_temp[col] = df_test[col].map(target_mean_dict)

  df_train_temp.drop(['survived'], axis=1, inplace=True)
  df_test_temp.drop(['survived'], axis=1, inplace=True)

  return df_train_temp, df_test_temp

categorical_vars = [col for col in data.columns if data[col].dtypes == 'O']

X_train_enc, X_test_enc = mean_encoding(X_train, X_test, categorical_vars)

from sklearn.metrics import roc_auc_score

roc_values = []

for feature in categorical_vars:

  roc_values.append(roc_auc_score(y_test.fillna(0), X_test_enc[feature].fillna(0)))
  
m1 = pd.Series(roc_values)
m1.index = categorical_vars
m1.sort_values(ascending=False) #features with roc_auc score greater or equal to 0.5 have significant importance on the prediction of the target

X_train['age_binned'], intervals = pd.qcut(
    X_train['age'],
    q=5,
    labels=False,
    retbins=True,
    precision=3,
    duplicates='drop'
) #qcut = same amount of observations per interval

X_test['age_binned'] = pd.cut(x = X_test['age'], bins=intervals, labels=False)
#pd.cut returns the discretised values which are "learnt" from the intervals built previously

numerical_vars = categorical_vars = [col for col in data.columns if data[col].dtypes != 'O']

for num_var in numerical_vars:

      X_train[num_var + '_binned'], intervals = pd.qcut(
          X_train['age'],
          q=5,
          labels=False,
          retbins=True,
          precision=3,
          duplicates='drop'
      ) 
      X_test[num_var + '_binned'] = pd.cut(x = X_test['age'], bins=intervals, labels=False)
   
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

binned_vars = [col for col in X_train.columns if 'binned' in col]
X_train_enc, X_test_enc = mean_encoding(
    X_train[binned_vars+['survived']], X_test[binned_vars+['survived']], binned_vars)

roc_values = []

for feature in binned_vars:

  roc_values.append(roc_auc_score(y_test.fillna(0), X_test_enc[feature]))
  
m1 = pd.Series(roc_values)
m1.index = binned_vars
m1.sort_values(ascending=False)
