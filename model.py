# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn


def rmsle_cv(model, train_x, train_y):
    n_folds = 5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x)
    rmse= np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def base_model():
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    nn = MLPRegressor(
    hidden_layer_sizes = (90, 90),
    alpha = 2.75
    )

    return ENet,lasso,nn

def xgb_Regressor(train_x, train_y, test_x):
    xgb_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
    scores = rmsle_cv(xgb_model, train_x, train_y)  
    xgb_model.fit(train_x, train_y)
    pred_train = xgb_model.predict(train_x)
    score = rmsle(train_y, pred_train)
    
    return xgb_model, scores.mean(), score, pred_train

def lgb_Regressor(train_x, train_y, test_x):
    lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 30)
    scores = rmsle_cv(lgb_model, train_x, train_y)  
    lgb_model.fit(train_x, train_y)
    pred_train = lgb_model.predict(train_x)
    score = rmsle(train_y, pred_train)
    
    return lgb_model, scores.mean(), score, pred_train

def gboost_Regressor(train_x, train_y, test_x):
    gb_model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=20, min_samples_split=20, 
                    loss='huber', random_state =5)
    scores = rmsle_cv(gb_model, train_x, train_y)  
    gb_model.fit(train_x, train_y)
    pred_train = gb_model.predict(train_x)
    score = rmsle(train_y, pred_train)
    
    return gb_model, scores.mean(), score, pred_train

def rt_Regressor(train_x, train_y, test_x):
    rt_model = RandomForestRegressor(n_estimators=1000,criterion='mse',
                          max_depth=5,max_features=20,min_samples_leaf=8,
                          n_jobs=12,random_state=17)#min_samples_leaf: 5~10
    scores = rmsle_cv(rt_model, train_x, train_y)  
    rt_model.fit(train_x, train_y)
    pred_train = rt_model.predict(train_x)
    score = rmsle(train_y, pred_train)
    
    return rt_model, scores.mean(), score, pred_train

def main():
    #load dataset
    train = pd.read_csv('cleaned_train20180222_000704.csv')
    test = pd.read_csv('cleaned_test20180222_000704.csv')
    #split dataset
    train_x = train.drop(['sale_quantity', 'sale_date'], axis=1)
    train_y = train['sale_quantity']
    test_x = test.drop(['sale_date', 'sale_quantity'], axis=1)
    all_data = pd.concat((train_x, test_x)).reset_index(drop=True)
    
    # mini, maxi = train_y.min(), train_y.max()
    # train_y = (train_y - mini) / (maxi - mini)

    #Dummy variables
    dummy_variables = ['compartment', 'type_id', 'level_id', 'department_id', 'TR',
                       'gearbox_type', 'driven_type_id', 'driven_type_id',
                       'emission_standards_id', 'cylinder_number', 'rated_passenger',
                       'sale_month', 'sale_year']
    for var in dummy_variables:
        all_data[var] = all_data[var].astype(object)
    all_data = pd.get_dummies(all_data)

    #Scale variables
    scale_variables = ['displacement', 'power', 'engine_torque', 'car_length',
                       'car_width', 'car_height', 'total_quality', 'wheelbase', 'front_track']
    scaler = MinMaxScaler()
    scaler.fit(train[scale_variables].values)
    all_data[scale_variables] = scaler.transform(all_data[scale_variables])
   
    train_x = all_data[:len(train_x)]
    test_x = all_data[len(train_x):]
    
    #开始训练
    print("XGBOOSTRegressor开始训练...")
    xgb_model, scores, score, xgb_train_pred = xgb_Regressor(train_x, train_y, test_x)
    print(scores, score)
    xgb_pred = xgb_model.predict(test_x)
    
    print("LGBMRegressor开始训练...")
    lgb_model, scores, score, lgb_train_pred= lgb_Regressor(train_x, train_y, test_x)
    print(scores, score)
    lgb_pred = lgb_model.predict(test_x)

    print("GDBTRegressor开始训练...")
    gb_model, scores, score, gb_train_pred = gboost_Regressor(train_x, train_y, test_x)
    print(scores, score)
    gb_pred = gb_model.predict(test_x)

    print("RandomForestRegressor开始训练...")
    rt_model, scores, score, rt_train_pred = rt_Regressor(train_x, train_y, test_x)
    print(scores, score)
    rt_pred = rt_model.predict(test_x)

    ENet,lasso,nn = base_model()
    
    ENet.fit(train_x, train_y)
    enet_pred = ENet.predict(train_x)
    print("ENet:", rmsle(train_y, enet_pred))
    
    lasso.fit(train_x, train_y)
    lasso_pred = lasso.predict(train_x)
    print("lasso:", rmsle(train_y, lasso_pred))
    
    nn.fit(train_x, train_y)
    nn_pred = nn.predict(train_x)
    print("nn:", rmsle(train_y, nn_pred))
    
    # ####Stacking####
    print('Stacking...')
    stacked_averaged_models = StackingRegressor(
    regressors=[ nn, ENet,lasso, lgb_model, gb_model],
    meta_regressor= xgb_model
    )
    stacked_averaged_models.fit(train_x, train_y)
    stacked_train_pred = stacked_averaged_models.predict(train_x.values)
    stacked_pred = stacked_averaged_models.predict(test_x.values)
    print(rmsle(train_y, stacked_train_pred))
    print(rmsle(train_y,stacked_train_pred*0.60 + gb_train_pred*0.10 + 
         rt_train_pred*0.10 + lgb_train_pred*0.10 + xgb_train_pred*0.10))
    ensemble = stacked_pred*0.60 + gb_pred*0.10 + rt_pred*0.10 + lgb_pred*0.10 + xgb_pred*0.10
    
    # #submission
    # sub = pd.read_csv("./data/test_20171225.csv")
    # sub['predict_quantity'] = ensemble
    # sub.to_csv('submission_ensemble.csv',index=False)
    

if __name__ == '__main__':
    main()