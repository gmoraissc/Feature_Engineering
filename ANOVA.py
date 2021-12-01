# test the hypothesis that 2 or more samples have the same mean

#samples are independent
#samples are normally distributed
#homogeneity of variance

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import f_classif, SelectKBest, SelectPercentile

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

univariate = f_classif(X_train.fillna(0), y_train)
univariate = pd.Series(univariate[1])
univariate.index = X_train.columns
univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ = SelectKBest(f_classif, k=3).fit(X_train.fillna(0), y_train)
X_train.columns[sel_.get_support()]
Index(['pclass', 'sex', 'embarked'], dtype='object')

X_train = sel_.transform(X_train.fillna(0))
X_train.shape
(914, 3)

data2 = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data2.select_dtypes(include=numerics).columns)
data2 = data2[numerical_vars]
data2.shape
(50000, 301)

x = data2.loc[:, data2.columns != 'target']
y = data2.loc[:, data2.columns == 'target']

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=0
)

from sklearn.feature_selection import f_regression

univariate = f_regression(X_train.fillna(0), y_train)
univariate = pd.Series(univariate[1])
univariate.index = X_train.columns
univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))

sel_ = SelectPercentile(f_regression, percentile=10).fit(X_train.fillna(0), y_train)
X_train.columns[sel_.get_support()]

from sklearn.feature_selection import VarianceThreshold, f_classif, SelectKBest

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

data = pd.read_csv(f'/content/drive/MyDrive/DS PROJECTS/Feature Selection/datasets/precleaned-datasets.zip (Unzipped Files)/dataset_1.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

X_train, X_test, y_train, y_test = train_test_split(
                                                    data.drop(labels=['target'], axis=1),
                                                    data['target'],
                                                    test_size=0.3,
                                                    random_state=0)

X_train_original = X_train.copy() #used to compare the ml perfomances
X_test_original = X_test.copy()

constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]

X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

sel = VarianceThreshold(threshold=0.01)

sel.fit(X_train)

features_to_keep = X_train.columns[sel.get_support()]

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep

X_test= pd.DataFrame(X_test)
X_test.columns = features_to_keep

duplicated_feat = []
for i in range(0, len(X_train.columns)):

    col_1 = X_train.columns[i]

    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)

X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

X_train_basic_filter = X_train.copy()
X_test_basic_filter = X_test.copy()

def correlation(dataset, threshold):
    
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested in absolute coeff value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    
    return col_corr


corr_features = correlation(X_train, 0.8)

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X_train_corr = X_train.copy()
X_test_corr = X_test.copy()

#ANOVA
sel_ = SelectKBest(f_classif, k=20).fit(X_train, y_train)

# capture selected feature names
features_to_keep = X_train.columns[sel_.get_support()]

# select features
X_train_anova = sel_.transform(X_train)
X_test_anova = sel_.transform(X_test)

# numpy array to dataframe
X_train_anova = pd.DataFrame(X_train_anova)
X_train_anova.columns = features_to_keep

X_test_anova = pd.DataFrame(X_test_anova)
X_test_anova.columns = features_to_keep


def run_randomForests(X_train, X_test, y_train, y_test):
    
    rf = RandomForestClassifier(n_estimators=200, random_state=39, max_depth=4)
    rf.fit(X_train, y_train)
    
    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))



def run_logistic(X_train, X_test, y_train, y_test):
    
    scaler = StandardScaler().fit(X_train)
    
    # function to train and test the performance of logistic regression
    logit = LogisticRegression(penalty='l1', random_state=44, max_iter=1000, solver='liblinear')
    logit.fit(X_train, y_train)
    
    print('Train set')
    pred = logit.predict_proba(scaler.transform(X_train))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = logit.predict_proba(scaler.transform(X_test))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
    
 run_randomForests(X_train_original,
                  X_test_original,
                  y_train, y_test)
Train set
Random Forests roc-auc: 0.807612232524249
Test set
Random Forests roc-auc: 0.7868832427636059
  
run_randomForests(X_train_basic_filter,
                  X_test_basic_filter,
                  y_train, y_test)
Train set
Random Forests roc-auc: 0.810290026780428
Test set
Random Forests roc-auc: 0.7914020645941601
  
run_randomForests(X_train_corr,
                  X_test_corr,
                  y_train, y_test)
Train set
Random Forests roc-auc: 0.8066004772684517
Test set
Random Forests roc-auc: 0.7859521124929707
  
run_randomForests(X_train_anova,
                  X_test_anova,
                  y_train, y_test)
Train set
Random Forests roc-auc: 0.8181634778452822
Test set
Random Forests roc-auc: 0.7994720109870546
  
# applying such methods have shown to have minor effects on model accuracy whilst removing substantially the feature space!

run_logistic(X_train_original,
             X_test_original,
             y_train, y_test)
Train set
Logistic Regression roc-auc: 0.7430426412785165
Test set
Logistic Regression roc-auc: 0.7514165331434336
  

run_logistic(X_train_basic_filter,
             X_test_basic_filter,
             y_train, y_test)
Train set
Logistic Regression roc-auc: 0.7410468829538979
Test set
Logistic Regression roc-auc: 0.7489081614486635
  
run_logistic(X_train_corr,
             X_test_corr,
             y_train, y_test)
Train set
Logistic Regression roc-auc: 0.7307283864065812
Test set
Logistic Regression roc-auc: 0.7227227435986561
  
run_logistic(X_train_anova,
             X_test_anova,
             y_train, y_test)
Train set
Logistic Regression roc-auc: 0.7385311277520487
Test set
Logistic Regression roc-auc: 0.7256599156189685
