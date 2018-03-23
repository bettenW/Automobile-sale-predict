# -*- coding: utf-8 -*-
import re
import time
import datetime
import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler

import warnings

def ignore_warn(*args ,**kwargs):
    pass
warnings.warn = ignore_warn

train = pd.read_csv("./data/train_20171226.csv")
test = pd.read_csv("./data/test_20171225.csv")
print(train.shape,test.shape)

#异常值处理
train['level_id'][train['level_id']=='-'] = '0'
#TR
train['TR'][train['TR']=='8;7']='8'
train['TR'][train['TR']=='5;4']='5'
#engine_torque
train['engine_torque'][train['engine_torque']=='-'] = '210'
train['engine_torque'][train['engine_torque']=='155/140'] = '155'
#fuel_type_id
train['fuel_type_id'][train['fuel_type_id']=='-']=1
#power
train['power'][train['power']=='81/70'] = '81'
#rated_passenger
train['rated_passenger'][train['rated_passenger']=='4月5日'] = '5'
train['rated_passenger'][train['rated_passenger']=='5月7日'] = '7'
train['rated_passenger'][train['rated_passenger']=='5月8日'] = '7'
train['rated_passenger'][train['rated_passenger']=='9'] = '7'
train['rated_passenger'][train['rated_passenger']=='6月7日'] = '6'
train['rated_passenger'][train['rated_passenger']=='6月8日'] = '6'
train['rated_passenger'][train['rated_passenger']=='7月8日'] = '8'

#转换类型
train['TR']=train['TR'].astype(int)
train['fuel_type_id']=train['fuel_type_id'].astype(int)
train['power']=train['power'].astype(float)
train['engine_torque']=train['engine_torque'].astype(float)
train['rated_passenger']=train['rated_passenger'].astype(int)
train['level_id']=train['level_id'].astype(int)

#定性特征转换
train['if_charging'] = train['if_charging'].map({'L':0, 'T':1})
train['if_MPV_id'] = train['if_MPV_id'].map({1:0, 2:1})
train['if_luxurious_id'] = train['if_luxurious_id'].map({1:0, 2:1})
train['newenergy_type_id'] = train['newenergy_type_id'].map({1:1, 2:2, 3:3, 4:2})
title_mapping = {"5WL": 1, "5-8W": 2, "8-10W": 3, "10-15W": 4, "15-20W": 5, "20-25W": 6, "25-35W": 7, "35-50W": 8, "50-75W": 9,}
train['price_level'] = train['price_level'].map(title_mapping)

#移除特征
train = train.drop(['newenergy_type_id', 'rear_track', 'equipment_quality', 'price'], axis=1)

#LabelEncoder
le_feat = ['brand_id', 'gearbox_type']
lbl = LabelEncoder()
for feat in le_feat:
    lbl.fit(list(train[feat])) 
    train[feat] = lbl.transform(train[feat])    
#sale_month, sale_year
train['sale_month'] = train['sale_date']%100
train['sale_year'] = train['sale_date']//100%100
test['sale_month'] = test['predict_date']%100
test['sale_year'] = test['predict_date']//100%100

'''
合并训练集，填充测试集
'''
#merge train dataset and fill test dataset
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

#找出销量最高的特征进行填充
cols = merge_train.columns.tolist()
test = test.ix[:,cols]
test['sale_date'] = test['sale_date'].fillna(201711)
test['sale_date'] = test['sale_date'].astype(int)
drop_col = ['sale_date', 'class_id', 'sale_quantity', 'sale_month', 'sale_year']
for co in drop_col:
    cols.remove(co)
all_class_id = test['class_id'].unique().tolist()
for idx in all_class_id:
    for c in cols:
        feat_data = train[[c, 'sale_quantity']][train['class_id']==idx].groupby([c], as_index=False).sum()
        max_val = feat_data['sale_quantity'].max()
        test[c][test['class_id']==idx] = list(feat_data[c][feat_data['sale_quantity']==max_val])[0]

#存储数据
print('new train shape:', merge_train.shape)
merge_train.to_csv(r'cleaned_train{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
print('new test shape:', test.shape)
test.to_csv(r'cleaned_test{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, float_format='%.4f')
