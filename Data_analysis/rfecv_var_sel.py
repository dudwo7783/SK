import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from os.path import join, os
from scipy import stats
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import RFECV, SelectKBest
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import time as tm
from sklearn.metrics import mean_squared_error
import _pickle as cPickle

# RTDB 경로
RTDB_path = r'C:\Users\Administrator\Desktop\KYJ\coke\regen_feature_tune_data\RTDB.csv'
data_dir = r'C:\Users\Administrator\Desktop\KYJ\coke\2nd_bz_anal_data'

def getDataFrame(path):
    df = pd.read_csv(path, index_col=0)
    return df

def strtodate(date_arr):
    x = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M').date() for d in date_arr]
    return x

def RemoveLowerOut(df,column_name):
    deout_df = df[(stats.zscore(df[column_name]) > -3)]
    return deout_df



if __name__ == "__main__":
    RTDB = getDataFrame(RTDB_path)
    RTDB = RTDB.drop(['Catalyst_V3_front_cl Content', 'Catalyst_V3_back_cl Content'], 1)

    # What is the more userful?

    date = strtodate(RTDB.index)
    max_2nd_bz = RTDB['max_2nd.Burning.Zone']
    # max_2nd_bz = np.array(RTDB['max_2nd.Burning.Zone'])

    RTDB = RemoveLowerOut(RTDB, 'max_2nd.Burning.Zone')

    date = strtodate(RTDB.index)
    max_2nd_bz = RTDB['max_2nd.Burning.Zone']

    # DrawTimePlot(date,max_2nd_bz, 'Remove outlier max temp plot')

    idx_RTDB = RTDB.reset_index()  # Remove NaN row

    x_RTDB = idx_RTDB.drop(['max_2nd.Burning.Zone', 'max_2nd.1dan', 'max_2nd.2dan',
                            'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'], 1)
    idx_RTDB = idx_RTDB.drop(['max_2nd.1dan', 'max_2nd.2dan',
                              'max_2nd.3dan', 'max_2nd.4dan', 'max_2nd.5dan', 'index'], 1)

    x_RTDB = x_RTDB[:len(x_RTDB) - 120]
    idx_RTDB = idx_RTDB[:len(idx_RTDB) - 120]

    train_x = x_RTDB.loc[1:(len(idx_RTDB) * 4 / 5)]
    train_y = idx_RTDB.loc[1:(len(idx_RTDB) * 4 / 5), ['max_2nd.Burning.Zone']]

    test_x = x_RTDB.loc[(len(idx_RTDB) * 4 / 5):]
    test_y = idx_RTDB.loc[(len(idx_RTDB) * 4 / 5):, ['max_2nd.Burning.Zone']]

    scaler = preprocessing.StandardScaler().fit(train_x)
    scale_train = scaler.transform(train_x)
    scale_test = scaler.transform(test_x)

    scaled_train_df = pd.DataFrame(
        scale_train, index=train_x.index, columns=train_x.columns)

    scaled_test_df = pd.DataFrame(
        scale_test, index=test_x.index, columns=test_x.columns)


    start_time = tm.time()
    estimator_rfe = SVR(kernel="linear", verbose=True)
    selector_rfe = RFECV(estimator_rfe, step=1, cv=StratifiedKFold(3), verbose=10, n_jobs=3)
    selector_rfe.fit(scaled_train_df, train_y)
    selector_rfe.support_
    print("svr rfe termination %s seconds ---" % (tm.time() - start_time))

    with open('real_rfecv.pkl', 'wb') as fid:
        cPickle.dump(selector_rfe, fid)

    print(train_x.columns.values[selector_rfe.support_ ])

    prediction = selector_rfe.predict(scaled_test_df)
    with open('prediction.pkl', 'wb') as fid:
        cPickle.dump(prediction, fid)
    real = test_y['max_2nd.Burning.Zone']
    score = selector_rfe.score(scaled_test_df, real)
    print(prediction)
    print(score)
    RMSE = mean_squared_error(real, prediction) ** 0.5
    print(RMSE)

    selector_rfe.predict(scale_test_df)
