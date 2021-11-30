import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import chi2, SelectKBest, SelectPercentile

data = pd.read_csv(f'https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
data.drop(labels = ['name','boat', 'ticket','body', 'home.dest'], axis=1, inplace=True)
data = data.replace('?', np.nan)
data.dropna(subset=['embarked', 'fare'], inplace=True)
data['age'] = data['age'].astype('float')
data['age'] = data['age'].fillna(data['age'].mean())
def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return 'N' 
data['cabin'] = data['cabin'].apply(get_first_cabin)
data['sex'] = np.where(data.sex == 'male', 1, 0)
ordinal_label = {k: i for i, k in enumerate(data['embarked'].unique(),0)}
data['embarked'] = data['embarked'].map(ordinal_label)

X_train, X_test, y_train, y_test = train_test_split(
    data[['pclass', 'sex', 'embarked']],
    data['survived'],
    test_size=0.3,
    random_state=0
)

f_score = chi2(X_train.fillna(0), y_train)
pvalues = pd.Series(f_score[1])
pvalues.index = X_train.columns
pvalues.sort_values(ascending=False)

# the bigger the sample size, the more small pvalues may contain, however]
# it does not necessarily indicates the feature is highly predictive (given
# that f_score ranks from smaller p_value to the biggest)
