REGRESSION COEFICIENTS

The predictors coefficients are directly proportional to how much the feature contributes to the final value of y

Coefficients in linear models depends on the following assumptions:

linear relationship between predictor X and outcome Y
Xs are independent
no-multicollinearity between Xs
Xs are normally distributed
Xs should be in the same scale
LOGISTIC REGRESSION COEFFICIENTS Predicts a quantitative response Y on the basis of a different predictor variable X.

Assumptions:

linear relationship between X and Y
coefficients magnitude are directly influenced by features scale
multivariate normality (X should follow a gaussian distribution/normally distributed)
no or little multicollinearity
homoscedasticity (variance should be the same/ homogeneity of variance = error term (noise) is the same across all values of the independent variables)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

Feature selection must be done after data pre-processing, which means categorical variables are encoded into numbers and therefore one can assess how deterministic they are of the target.

To avoid overfitting, all feature selection procedures should be done by examining only the training set.

data = pd.read_csv(f'/content/drive/MyDrive/data science/Datasets/paribas_train.csv')
data.shape
(114321, 133) #133 variables (131 as possible predictors, given that there is the column 'id' and the target

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]

data.shape
(114321, 114) #in this demo, only the numerical variables were used. If it was to be used the entire dataset, one would have to encode categorical into numercial

X = data.drop(labels=['target', 'ID'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

X_train = X_train.fillna(0)

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

Specify the logistic regression model. Select ridge penalty (L2) (which is default).

Thus evaluating coefficients magnitude (not wether Lasso shrinks coeff to zero)

Idea is to avoid regularisation so the coeffs are not affected/modified by the regularisation penalty (to do this, one must set a big C parameter eg=1000, as to fit a non regularised logistic regression)

Then SelectFromModel automatically select the features

sel_ = SelectFromModel(LogisticRegression(C=1000, penalty='l2', max_iter=10000))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

Selects features which coeff values are greater than the mean of all coeffs (compares absolute values of coefficients)

selected_feat = X_train.columns[(sel_.get_support())]
pd.Series(sel_.estimator_.coef_.ravel()).hist()

pd.Series(np.abs(sel_.estimator_.coef_).ravel()).hist()

SELECT FEATURES BASED ON REGRESSION COEFFICIENTS IN LINEAR MODELS

from sklearn.linear_model import LinearRegression

data = pd.read_csv(f'/content/sample_data/california_housing_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(LinearRegression()) #LR from sklearn is a non-regularised linear method which fits by matrix multiplication and not by gradient descent
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

selected_feat = X_train.columns[(sel_.get_support())]

pd.Series(np.abs(sel_.estimator_.coef_).ravel()).hist(bins=50)

THE EFFECTS OF REGULARISATION IN REGRESSION COEFFICIENTS

Regularisation applies a penalty on the coefficients in order to reduce their influence and create models that generalise better.

Although it improves model performance, the true relationship between the predicto X and the outcome Y is masked.

data = pd.read_csv(f'/content/sample_data/california_housing_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)
scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

coefs_df = []

for c in [1, 10, 100, 1000]:
  logit = LogisticRegression(C=c, max_iter=100000, penalty='l2') # c = inverse of the regularisation strenght, smaller c values = stronger regularisations
  logit.fit(scaler.transform(X_train.fillna(0)), y_train)

  coefs_df.append(pd.Series(logit.coef_.ravel()))
  
  coefs = pd.concat(coefs_df, axis=1)
coefs.columns = [1, 10, 100, 1000]
coefs.index = X_train.columns
coefs.head()

	1	10	100	1000
v1	-0.111814	-0.204570	-0.182473	-0.061949
v2	0.045496	0.046628	0.046214	0.045942
v4	0.057645	0.055533	0.055484	0.053563
v5	0.039291	0.046771	0.049687	0.047743
v6	0.101106	0.104815	0.104705	0.105287

coefs.columns = np.log([1, 10, 100, 1000])
coefs.head()

	0.000000	2.302585	4.605170	6.907755
v1	-0.111814	-0.204570	-0.182473	-0.061949
v2	0.045496	0.046628	0.046214	0.045942
v4	0.057645	0.055533	0.055484	0.053563
v5	0.039291	0.046771	0.049687	0.047743
v6	0.101106	0.104815	0.104705	0.105287

coefs.T.plot(figsize=(15,10)) #the coefficients change with regualrisation intensity

____

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score

data = pd.read_csv(f'/content/sample_data/california_housing_train.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

X_train_original = X_train.copy()
X_test_original = X_test.copy()

REMOVING CONSTANT FEATURES
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]

X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)

REMOVING QUASI-CONSTANT FEATURES
sel = VarianceThreshold(threshold=0.01)
sel.fit(X_train)
features_to_keep = X_train.columns[sel.get_support()]
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep

X_test= pd.DataFrame(X_test)
X_test.columns = features_to_keep

REMOVING DUPLICATE FEATURES
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

REMOVING CORRELATED FEATURES

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)

X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

X_train_corr = X_train.copy()
X_test_corr = X_test.copy()

SELECTING FEATURES BY THE REGRESSION COEFFICIENTS

scaler = StandardScaler()
scaler.fit(X_train)

sel_ = SelectFromModel(
    LogisticRegression(C=0.0005, random_state=10, max_iter=10000, penalty='l2'))

#indices_to_keep = ~X_train.isin([np.nan, np.inf, -np.inf]).any(1)
#X_train = X_train[indices_to_keep].astype(np.float64)

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
y_train.replace([np.inf, -np.inf], np.nan, inplace=True)

sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

X_train_coef = pd.DataFrame(sel_.transform(X_train.fillna(0)))
X_test_coef = pd.DataFrame(sel_.transform(X_test.fillna(0)))

X_train_coef.columns = X_train.columns[(sel_.get_support())]
X_test_coef.columns = X_train.columns[(sel_.get_support())]

COMPARING PERFORMANCE

def run_logistic(X_train, X_test, y_train, y_test):
    
    scaler = StandardScaler().fit(X_train)
    
    logit = LogisticRegression(C=0.0005, random_state=10, max_iter=10000, penalty='l2')
    logit.fit(scaler.transform(X_train), y_train)
    
    print('Train set')
    pred = logit.predict_proba(scaler.transform(X_train))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = logit.predict_proba(scaler.transform(X_test))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
    
    # original dataset - all variables
run_logistic(X_train_original.fillna(0),
             X_test_original.fillna(0),
             y_train.fillna(0),
             y_test.fillna(0))
Train set
Logistic Regression roc-auc: 0.7055919341080461
Test set
Logistic Regression roc-auc: 0.7007011331591052
  
  # filter methods - basic
run_logistic(X_train_basic_filter.fillna(0),
             X_test_basic_filter.fillna(0),
             y_train.fillna(0),
             y_test.fillna(0))
Train set
Logistic Regression roc-auc: 0.7055919341080461
Test set
Logistic Regression roc-auc: 0.7007011331591052
  
  # filter methods - correlation
run_logistic(X_train_corr.fillna(0),
             X_test_corr.fillna(0),
             y_train.fillna(0),
             y_test.fillna(0))

Train set
Logistic Regression roc-auc: 0.7052155246389926
Test set
Logistic Regression roc-auc: 0.700441833087615
  
  # embedded methods - Logistic regression coefficients
run_logistic(X_train_coef,
             X_test_coef,
             y_train,
             y_test)

Train set
Logistic Regression roc-auc: 0.704719836887429
Test set
Logistic Regression roc-auc: 0.7007974252990639
  
  #A model with least features performs as well as a model with all the variables
