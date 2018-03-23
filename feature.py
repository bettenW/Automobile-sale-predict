# -*- coding: utf-8 -*-

import time
import datetime
import numpy as np
import pandas as pd
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.preprocessing import PolynomialFeatures

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

#读取数据
init_train = pd.read_csv("./data/train_20171226.csv")
train = pd.read_csv("cleaned_train20180203_181856.csv")
test = pd.read_csv("./data/test_20171225.csv")

# print("price_level拆分为最低价,中间价,最高价...")
#price_level拆分为最低价,中间价,最高价
price_level = init_train[['price_level', 'class_id']]
price_level['low_price'] = price_level['price_level'].map({'5WL':3, '5-8W':5, '8-10W':8, 
                            '10-15W':10, '15-20W':15, '20-25W':20, '25-35W':25, '35-50W':35, '50-75W':50})
price_level['mean_price'] = price_level['price_level'].map({'5WL':4, '5-8W':7.5, '8-10W':9, 
                            '10-15W':12.5, '15-20W':17.5, '20-25W':22.5, '25-35W':30, '35-50W':42.5, '50-75W':62.5})
price_level['high_price'] = price_level['price_level'].map({'5WL':5, '5-8W':8, '8-10W':10, 
                            '10-15W':15, '15-20W':20, '20-25W':25, '25-35W':35, '35-50W':50, '50-75W':75})
train['low_price'] = price_level['low_price']
train['mean_price'] = price_level['mean_price']
train['high_price'] = price_level['high_price']
#添加discount_price, raise_price
train['discount_price'] = 0
train['discount_price'][train['price']<train['low_price']] = 1
train['raise_price'] = 0
train['raise_price'][train['price']>train['high_price']] = 1

#sale_date拆分为年,月,对月进行one-hot
train['sale_month'] = train['sale_date']%100
train['sale_year'] = train['sale_date']//100%100
test['sale_month'] = test['predict_date']%100
test['sale_year'] = test['predict_date']//100%100

print('添加时间特征latest和passed...')
#添加时间特征latest和passed
train['latest'] = 0
train['passed'] = 0
for i in range(train['class_id'].count()):
    id = train['class_id'][i]
    min_date = train["sale_date"][train['class_id']==id].min()
    max_date = train["sale_date"][train['class_id']==id].max()
    loc_date = train["sale_date"][i]
    train["latest"][i] = (loc_date//100%100- min_date//100%100)*12+(loc_date%100- min_date%100)
    train["passed"][i] = (max_date//100%100- loc_date//100%100)*12+(max_date%100- loc_date%100)
test['latest'] = 0
test['passed'] = 0
for i in range(test['class_id'].count()):
    id = test['class_id'][i]
    min_date = train["sale_date"][train['class_id']==id].min()
    max_date = test["predict_date"][test['class_id']==id].max()
    loc_date = test["predict_date"][i]
    test["latest"][i] = (loc_date//100%100- min_date//100%100)*12+(loc_date%100- min_date%100)
    test["passed"][i] = (max_date//100%100- loc_date//100%100)*12+(max_date%100- loc_date%100)

print("开始合并train集...")
#合并数据集,同一class_id每月只有一次,依据为销量最高
merge_train = pd.DataFrame()
all_class_id = train['class_id'].unique().tolist()
for idx in all_class_id:
    all_month = train['sale_date'][train['class_id']==idx].unique().tolist()
    for mon in all_month:
        max_val = train['sale_quantity'][train['class_id']==idx][train['sale_date']==mon].max()
        sale_sum = train['sale_quantity'][train['class_id']==idx][train['sale_date']==mon].sum()
        feat_val = train[train['class_id']==idx][train['sale_date']==mon][train['sale_quantity']==max_val]
        feat_val['sale_quantity'] = sale_sum
        merge_train = pd.concat([merge_train, feat_val[0:1]])
merge_train = merge_train.sort_values(by=['sale_date'], ascending=True)

print("开始填充test集...")
#填充test datasets 特征
cols = merge_train.columns.tolist()
test = test.ix[:,cols]
test['sale_date'] = test['sale_date'].fillna(201711)
test['sale_date'] = test['sale_date'].astype(int)

drop_col = ['sale_date', 'class_id', 'sale_quantity', 'sale_month', 'sale_year', 'latest', 'passed']
for co in drop_col:
    cols.remove(co)

#找出销量最高的特征进行填充
all_class_id = test['class_id'].unique().tolist()
for idx in all_class_id:
    for c in cols:
        feat_data = train[[c, 'sale_quantity']][train['class_id']==idx].groupby([c], as_index=False).sum()
        max_val = feat_data['sale_quantity'].max()
        test[c][test['class_id']==idx] = list(feat_data[c][feat_data['sale_quantity']==max_val])[0]

#单独填充price,discount_price,rsise_price
for idx in all_class_id:
    feat_data = train[['price', 'sale_quantity']][train['class_id']==idx].groupby(['price'], as_index=False).sum()
    feat_data['total_price'] = feat_data['price']*feat_data['sale_quantity']
    total_sale = feat_data['sale_quantity'].sum()
    mean_price = feat_data['total_price'].sum()/total_sale
    test['price'][test['class_id']==idx] = mean_price
test['discount_price'] = 0
test['discount_price'][test['price']<test['low_price']] = 1
train['raise_price'] = 0
test['raise_price'][test['price']>test['high_price']] = 1

#添加量差特征，即本月与上月销量差距
temp_train = merge_train.sort_values(['class_id', 'sale_date'])
temp_train['diff_quantity'] = np.nan
temp_train['growth_rate'] = np.nan
all_class_id = temp_train['class_id'].unique().tolist()
for idx in all_class_id:
    one_id = temp_train.loc[temp_train['class_id']==idx]
    all_date = one_id['sale_date'].unique().tolist()
    for i in range(len(all_date)-1):
        last = list(one_id['sale_quantity'].loc[one_id['sale_date']==all_date[i]])[0]
        now = list(one_id['sale_quantity'].loc[one_id['sale_date']==all_date[i+1]])[0]
        temp_train['diff_quantity'].loc[(temp_train['class_id']==idx)&(temp_train['sale_date']==all_date[i+1])] = now-last
        temp_train['growth_rate'].loc[(temp_train['class_id']==idx)&(temp_train['sale_date']==all_date[i+1])] = (now-last)/now
merge_train = temp_train.sort_index()

#预测训练集首月diff_quantity, growth_rated缺失值
import xgboost as xgb
xgboost_model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
xc = ['class_id']+['sale_quantity']+['brand_id']+['sale_month']+['sale_year']+['latest']+['passed']
xtrain = merge_train[merge_train['diff_quantity'].notnull()]
xtrain = xtrain[:][xc]
xtest = merge_train[merge_train['diff_quantity'].isnull()]
xtest = xtest[:][xc]
y_diff_train = merge_train['diff_quantity'][merge_train['diff_quantity'].notnull()]
y_grow_train = merge_train['growth_rate'][merge_train['diff_quantity'].notnull()]
xgboost_model.fit(xtrain, y_diff_train)
diff = xgboost_model.predict(xtest)
xgboost_model.fit(xtrain, y_grow_train)
grow = xgboost_model.predict(xtest)
xtest['diff_quantity'] = diff
xtest['growth_rate'] = grow
for idx in all_class_id:
    merge_train['diff_quantity'].loc[(merge_train['class_id']==idx)&(merge_train['diff_quantity'].isnull())]=xtest['diff_quantity'][xtest['class_id']==idx]
    merge_train['growth_rate'].loc[(merge_train['class_id']==idx)&(merge_train['growth_rate'].isnull())]=xtest['growth_rate'][xtest['class_id']==idx]

#填充测试集首月diff_quantity, growth_rated缺失值
test['diff_quantity'] = np.nan
test['growth_rate'] = np.nan
for idx in all_class_id:
    diff = merge_train['diff_quantity'].loc[(merge_train['class_id']==idx)&(merge_train['sale_month']==11)].mean()
    grow = merge_train['growth_rate'].loc[(merge_train['class_id']==idx)&(merge_train['sale_month']==11)].mean()
    test['diff_quantity'].loc[test['class_id']==idx] = diff
    test['growth_rate'].loc[test['class_id']==idx] = grow
test = test.fillna(0)
'''
之后考虑,依据时间权重,进行特征加权填充
'''
test = test.drop(['sale_quantity'], axis=1) 
merge_train.to_csv(r'feature_train{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
test.to_csv(r'feature_test{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')