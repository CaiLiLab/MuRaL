import sys

#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


import category_encoders as ce

import pandas as pd
import numpy as np

from evaluation import f3mer_comp, f5mer_comp, f7mer_comp

#from temperature_scaling import 

#get the scores for the trained model
def get_score(model, X, y, X_val, y_val):
    model.fit(X, y)
    y_pred = model.predict_proba(X_val)[:,1]
    auc_score = roc_auc_score(y_val, y_pred)
    brier_score = brier_score_loss(y_val, y_pred)
    return auc_score,brier_score

#set train file
train_file = sys.argv[1]
print('train file: ' + train_file)

#set test file
test_file = sys.argv[2]
print('test file: ' + test_file)

#
n_iter = int(sys.argv[3])

#read the data
data = pd.read_csv(train_file).dropna()

data_test = pd.read_csv(test_file).dropna()

X_orig = data.drop(['mut_type'], axis=1)
y = data['mut_type']

#extract the category columns
cat_columns = data.columns.tolist()[0:11]

#con_idx = np.append([11], list(range(13,27)))

#con_columns = list( data.columns.tolist()[i] for i in con_idx)

#use the OneHotEncoder to transform the data
ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True, cols=cat_columns)

# 'sag' is very sensitive to feature scaling so standardizing our features is particularly important.
scaler = StandardScaler()

#X = scaler.fit_transform(ohe.fit_transform(X_orig))
X = ohe.fit_transform(X_orig)

#ohe.inverse_transform(data1).shape

#set train data, validation data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=101010)

X_orig_train, X_orig_val, y_train, y_val = train_test_split(X_orig, y, test_size=0.2, random_state=101010)

#set testing data
X_test_orig = data_test.drop(['mut_type'], axis=1)

y_test = data_test['mut_type']

#X_test = scaler.fit_transform(ohe.fit_transform(X_test_orig))
X_test = ohe.fit_transform(X_test_orig)


#Logistic Regression model; NOTE: 'lbfgs' is not suitable for large datasets!
#logit = LogisticRegression(C=0.1, class_weight='balanced', penalty='l1', solver='saga', max_iter=n_iter, random_state=666, tol=1e-4, verbose=0, n_jobs=3)
logit = LogisticRegression(C=10, penalty='l1', solver='saga', max_iter=n_iter, random_state=666, tol=1e-4, verbose=0, n_jobs=3)

#Random Forest model
#rf = RandomForestClassifier(n_estimators=100)


#baseline_logit_score = get_score(logit, X_train, y_train, X_val, y_val)
#baseline_rf_score = get_score(rf, X_train, y_train, X_val, y_val)

#The default scoring option used by LogisticRegressionCV is 'accuracy'
#NOTE: increase the max_iter if there is Covergence warning

#logit_model_cv = LogisticRegressionCV(cv=10, class_weight='balanced', penalty='l1', multi_class='auto', tol=1e-4, scoring="brier_score_loss", solver="saga", max_iter=n_iter,random_state=666, n_jobs=2)
#logit_model_cv = LogisticRegressionCV(cv=10, penalty='l2', tol=1e-8, scoring="neg_log_loss", solver="lbfgs", max_iter=5000)

#add interactions
#x_t = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(x)

#fit = logit_model_cv.fit(X, y)
fit = logit.fit(X, y)

print(fit)
#print(fit.C_)

#fit = logit.fit(X_train, y_train)


roc_auc_score(y_val, fit.predict_proba(X_val)[:,1])

brier_score_loss(y_train, fit.predict_proba(X_train)[:,1])

log_loss(y_train, fit.predict_proba(X_train))

#brier_score_loss(y_test, fit.predict_proba(X_test)[:,1])

brier_score_loss(y_val, fit.predict_proba(X_val)[:,1])

#pd.concat([y_val.reset_index()['mut_type'], pd.Series(data=fit.predict_proba(X_val)[:,1])], axis=1)

#X_out = ohe.inverse_transform(X_test).reset_index().drop(['index'], axis=1)
X_out = X_test_orig.reset_index().drop(['index'], axis=1)
y_out = y_test.reset_index().drop(['index'], axis=1)
y_prob = pd.Series(data=fit.predict_proba(X_test)[:,1], name="prob")
#y_prob1 = pd.Series(data=cross_val_predict(logit_model_cv, X_test, method='predict_proba')[:,1], name="prob1")
#print (fit.predict_proba(X_test))

#save the data along with the probabilities
#data_and_prob = pd.concat([data_test, y_prob], axis=1)
data_and_prob = pd.concat([X_out, y_out, y_prob], axis=1)
data_and_prob.to_csv('new_file.tsv', sep='\t', index=False, float_format='%.3f')

X_data = X_orig.reset_index().drop(['index'], axis=1)
y_data = y.reset_index().drop(['index'], axis=1)
y_prob_all = pd.Series(data=fit.predict_proba(X)[:,1], name="prob")
data_and_prob_all = pd.concat([X_data, y_data, y_prob_all], axis=1)

#print (fit.predict_proba(X))
#np.set_printoptions(threshold=sys.maxsize)

#print (data_and_prob[['us1','ds1','mut_type','prob']].groupby(['us1','ds1']).mean())

#def model_coef(x):
#   return 1 / (1 + np.exp(-x))

#model(x_test * clf.coef_ + clf.intercept_).ravel()
print(fit.coef_)

print ('3mer correlation - test: ' + str(f3mer_comp(data_and_prob)))
print ('3mer correlation - all: ' + str(f3mer_comp(data_and_prob_all)))
print ('5mer correlation - test: ' + str(f5mer_comp(data_and_prob)))
print ('5mer correlation - all: ' + str(f5mer_comp(data_and_prob_all)))
print ('7mer correlation - test: ' + str(f7mer_comp(data_and_prob)))
print ('7mer correlation - all: ' + str(f7mer_comp(data_and_prob_all)))

#print (data_and_prob[['us2','us1','ds1','mut_type','prob']].groupby(['us2', 'us1','ds1']).agg(['mean', 'sum', 'count']).to_string())

