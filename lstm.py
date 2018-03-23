# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import xgboost as xgb

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn


def rmsle_cv(model, train_x, train_y):
    n_folds = 10
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def xgb_Regressor(train_x, train_y, test_x):
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    scores = rmsle_cv(model_xgb, train_x, train_y)
    
    model_xgb.fit(train_x, train_y)
    pred_train = model_xgb.predict(train_x)
    score = rmsle(train_y, pred_train)
    
    return scores.mean(), score

def main():

    #load dataset
    train = pd.read_csv('feature_train20180201_195350.csv')
    test = pd.read_csv('feature_test20180201_195350.csv')

    
    # train_x = train.drop(['sale_quantity', 'sale_date'], axis=1)
    # train_y = train['sale_quantity']
    # test_x = test.drop(['sale_date'], axis=1)

    #类型转换
    cols = train_x.columns.tolist()
    for c in cols:
        test_x[c] = test_x[c].astype(train_x[c].dtypes)

    #对class_id LabelEncoder
    all_data = pd.concat((train_x, test_x)).reset_index(drop=True)
    all_data = all_data.ix[:,cols]
    lbl = LabelEncoder()
    lbl.fit(list(all_data['class_id'])) 
    all_data['class_id'] = lbl.transform(all_data['class_id'])
    #OneHotEncoder 月
    all_data['sale_month'] = all_data['sale_month'].astype(str)
    all_data = pd.get_dummies(all_data)
    
    #pca降维
    pca = PCA(n_components = 20)
    pca.fit(all_data)
    print(pca.explained_variance_ratio_)
    all_data = pca.transform(all_data)
    # new_data = pd.DataFrame(new_data)
    # all_data = pd.concat((all_data, new_data)).reset_index(drop=True)

    train_x = all_data[:len(train_x)]
    test_x = all_data[len(train_x):]

    print("xgboost model开始训练...")
    scores, score = xgb_Regressor(train_x, train_y, test_x)
    print(scores, score)
    #xgb_pred = model_xgb.predict(test_x)

    # sub = pd.read_csv("./data/test_20171225.csv")
    # sub['predict_quantity'] = xgb_pred
    # sub.to_csv('submission_xgb_16year.csv',index=False)
    

if __name__ == '__main__':
    main()