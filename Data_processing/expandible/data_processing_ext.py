from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np
from os.path import join, basename, splitext
import glob
import calendar
from datetime import date, time, timedelta
from datetime import datetime as dt


def mergeFile(data_dir, split_files, file):
    save_path = join(data_dir, file+".csv")
    df = pd.concat((pd.read_csv(f, header=None, index_col=0) for f in split_files))

    df.to_csv(save_path, header=False)

    return save_path

def getFileList(data_dir, split_files):
    allFiles = glob.glob(join(data_dir, "**/*.csv"), recursive=True)

    L_files = [f for f in allFiles if '.L' in f]
    L_tags = [splitext(basename(f))[0] for f in L_files]
    Y_files = [f for f in allFiles if '.L' not in f]
    Y_tags = [splitext(basename(f))[0] for f in Y_files]

    if len(split_files) != 0 :
        for file in split_files:
            L_files = [f for f in L_files if file not in f]
            L_tags = [f for f in L_tags if file not in f]
            Y_files = [f for f in Y_files if file not in f]
            Y_tags = [f for f in Y_tags if file not in f]

        for file in split_files:
            split_files = [f for f in allFiles if file in f]
            splitfile = mergeFile(data_dir, split_files, file)
            Y_files.append(splitfile)
            Y_tags.append(file)

    return L_files, L_tags, Y_files, Y_tags


def getDataFrame(L_files, L_tags, Y_files, Y_tags, time_unit, start_date, end_date, fill_method):

    # replace abbreviated name to number
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}


    Ldf = pd.concat((pd.read_csv(f, header=None) for f in L_files), keys=L_tags)
    Ydf = pd.concat((pd.read_csv(f, header=None) for f in Y_files), keys=Y_tags)

    Ldf = Ldf.rename(columns={0: 'date', 1: 'data'})
    Ydf = Ydf.rename(columns={0: 'date', 1: 'data'})
    Ldf.index.names = ['tag', 'idx']
    Ydf.index.names = ['tag', 'idx']

    Ldf['date'] = Ldf['date'].apply(lambda x: '0' + x if len(x) == 17 else x)
    Ydf['date'] = Ydf['date'].apply(lambda x: '0' + x if (len(x) == 17) or (len(x) == 21) else x)

    if time_unit == '1min':
        Ldf['date'] = Ldf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), 1)))

        Ydf['date'] = Ydf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), 1)))
    '''
    elif time_unit == 'S':
        Ldf['date'] = Ldf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), int(x[16:18]))))

        Ydf['date'] = Ydf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), 0)))


    elif time_unit == 'H':
        Ldf['date'] = Ldf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), int(x[16:18]))))

        Ydf['date'] = Ydf['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), 0, 0)))

    elif time_unit == 'D':
        Ldf['date'] = Ldf['date'].apply(lambda x: date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])))

        Ydf['date'] = Ydf['date'].apply(lambda x: date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])))
    '''


    start = date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))

    Ydf['data'] = Ydf['data'].apply(lambda x: np.nan if x == 'Digital State' else x)
    Ydf['data'].apply(lambda x: float(x))

    ix = pd.DatetimeIndex(start=start, end=end, freq=time_unit) + timedelta(seconds=1)

    if fill_method == 'pad_to_bfill' :
        Ldf = pd.concat((Ldf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='pad').fillna(method='bfill') for tag in L_tags), keys=L_tags)
        Ydf = pd.concat((Ydf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='pad').fillna(method='bfill')for tag in Y_tags), keys=Y_tags)

    '''
    elif fill_method == 'bfill_to_pad' :
        Ldf = pd.concat((Ldf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='bfill').fillna(method='pad') for tag in L_tags), keys=L_tags)
        Ydf = pd.concat((Ydf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='bfill').fillna(method='pad') for tag in Y_tags), keys=Y_tags)

    else :
        Ldf = pd.concat((Ldf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='pad').fillna(method='bfill') for tag in L_tags), keys=L_tags)
        Ydf = pd.concat((Ydf.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                        reset_index().fillna(method='pad').fillna(method='bfill')for tag in Y_tags), keys=Y_tags)

    '''

    Ldf = pd.concat((Ldf.loc[tag, "data"] for tag in L_tags), axis=1, keys=L_tags).set_index(ix).reset_index()
    Ldf.columns.values[0] = 'date'

    Ydf = pd.concat((Ydf.loc[tag, "data"] for tag in Y_tags), axis=1, keys=Y_tags).set_index(ix).reset_index()
    Ydf.columns.values[0] = 'date'

    RTDB = pd.concat([Ldf, Ydf], axis=1)
    RTDB = RTDB[:len(RTDB) - 1]

    return Ldf, Ydf, RTDB