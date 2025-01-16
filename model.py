import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from copy import copy
from time import time
from sklearn.preprocessing import LabelEncoder
from timeit import default_timer
import numpy as np
import xgboost as xgb

np.random.seed(30)

df = pd.read_csv("credit_card_transactions_with_fraud.csv")

#View Database 
print()
print(df.head())

#Check for how much fraudulent data we have currently
print()
print(df["Fraudulent"].value_counts())

#Encode the label so we could later map our classes to integers later
labe = LabelEncoder()
labe.fit(df.Fraudulent)
print()
print(df.dtypes)

category_values = ['Time','Merchant', 'Category']

category_data = pd.get_dummies(df[category_values])
print()
print(category_data.head())

numeric_vals = list(set(df.columns.values.tolist()) - set(category_values)).remove('Fraudelent')
numeric_data = df[numeric_vals].copy()

ncdata = pd.concat([numeric_data, category_data])

labels = df["Fraudulent"].copy()
integer_label = labe.transform(labels)

x_train, x_test, y_train, y_test = train_test_split(ncdata, integer_label, test_size=.15, random_state=60)

preprocessed_data = {
    'x_train':x_train,
    'x_test':x_test,
    'y_train':y_train,
    'y_test':y_test,
    'labe':labe
    
}

path = 'preprocessedDataFull.pkl'
f = open(path, 'wb')
pickle.dump(preprocessed_data, f)
f.close()

fraudidx =np.where(y_train.classes_ == 'Yes')[0][0]
binary_train = y_train.copy()
binary_train[y_train == fraudidx] = 1
binary_train[y_train != fraudidx] = 0

binary_test =np.where(y_test.classes_ == 'Yes')[0][0]
binary_test = y_test.copy()
binary_test[y_test == fraudidx] = 1
binary_test[y_test != fraudidx] = 0

print()
print('Number of anomalies in y_train: ', binary_train.sum())
print('Number of anomalies in y_train: ', binary_test.sum())

param = {
    'num_rounds': 15,
    'max_depth':0,
    'max_leaves':2**8,
    'alpha':0.9,
    'eta':0.1,
    'gamma':0.1,
    'learning_rate':0.1,
    'subsample':1,
    'reg_lambda':1,
    'scal_pos_weight':2,
    'tree_method': 'gpu_hist',
    'n_gpus':1,
    'objective':'binary:logistic',
    'verbose':True
}

dtrain = xgb.DMatrix(x_train, label=binary_train)
dtest = xgb.DMatrix(x_test, labels=binary_test)
evals = [(dtest, 'test'), (dtrain, 'train')]
num_rounds = param['num_rounds']
model = xgb.train(param, dtrain, num_rounds, evals=evals)


threshold = .5
true_labels = binary_test.astype(int)
preds = model.predict(dtest)
pred_labels = (preds > threshold).astype(int)
