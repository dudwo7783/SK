from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import pandas as pd
import numpy as np
from os.path import join, basename, splitext, exists
import glob
from datetime import date, time, timedelta
from datetime import datetime as dt
from multiprocessing import Process, Queue
import calendar

'''
 * 분할되어있는 태그 csv 파일을 하나의 파일로 합쳐 저장한다.
 * 파일을 불러 열로 concatenate 한다.
 * 파일 존재 시 경로만 반환
 * 없을 시 csv 파일 생성
'''
def mergeFile(data_dir, split_files, file):
    save_path = join(data_dir, file + ".csv")

    if exists(save_path):
        return save_path

    df = pd.concat((pd.read_csv(f, header=None, index_col=0) for f in split_files))
    df.to_csv(save_path, header=False)

    return save_path

'''
 * 지정한 root 디렉터리의 모든 csv 파일을 읽는다.
 * 파일 경로(files)와 태그(tags)를 따로 리스트에 저장한다.
 * files와 tags에서 분할 파일 이름이 포함되어있는 요소들은 제거한다
 * 분할된 파일들을 합치고, 합쳐진 파일과 태그 이름을 각 files와 tags에 추가한다.
'''
def getFileList(data_dir, split_files):
    allFiles = glob.glob(join(data_dir, "**/*.csv"), recursive=True)

    files = [f for f in allFiles]                       # Full path name
    tags = [splitext(basename(f))[0] for f in files]    # File name

    if len(split_files) != 0 :
        for file in split_files:    #Extract not split file
            files = [f for f in files if file not in f]
            tags = [f for f in tags if file not in f]

        for file in split_files:    #Extract split file and merge
            split_files = [f for f in allFiles if file in f]
            splitfile = mergeFile(data_dir, split_files, file)
            files.append(splitfile)
            tags.append(file)

    return files, tags

'''
 * 모든 파일들을 열로 길게 읽는다.(MultiIndex를 사용해 태그 구분)
 * String 타입으로 읽어들인 날짜를 Factor로 변환한다.
 * 'Digital State'를 값으로 갖는 요소들은 Nan으로 변환한다.
 * 날짜를 기준으로 중복 데이터의 첫번째 값만 남겨놓고
     1분 단위의 index에 매칭시킨다.
     매칭과정에서 발생한 Nan 요소들은 이전 최근의 값으로 대체한다.
 * 열을 행으로 transpose 한다.
'''

def getDataFrame(files, tags, ix, queue):

    # replace abbreviated name to number
    abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}

    df = pd.concat((pd.read_csv(f, header=None) for f in files), keys=tags)
    df = df.rename(columns={0: 'date', 1: 'data'})
    df.index.names = ['tag', 'idx']




    # Change "1-Jan-17" form to "01-Jan-17" form
    df['date'] = df['date'].apply(lambda x: '0' + x if (len(x) == 17) or (len(x) == 21) else x)

    # change string date to factor(all second is "1")
    df['date'] = df['date'].apply(lambda x: dt.combine(date(int("20" + x[7:9]), abbr_to_num[x[3:6]], int(x[:2])),
                                                             time(int(x[10:12]), int(x[13:15]), 0)))



    df['data'] = df['data'].apply(lambda x: np.nan if x == 'Digital State' else x)
    # change data type to float
    df['data'].apply(lambda x: float(x))



    # Pad Nan elements before bfill
    df = pd.concat((df.loc[tag,].drop_duplicates(subset='date', keep='first').set_index('date').reindex(ix).
                   reset_index().fillna(method='pad').fillna(method='bfill') for tag in tags), keys=tags)

    df = pd.concat((df.loc[tag, "data"] for tag in tags), axis=1, keys=tags).set_index(ix)
    df = df[:len(df) - 1]

    queue.put(df)


def parallel_processing(files, tags, start_date, end_date):
    print('parallel computing start')

    # The number of process
    n_jobs = 8

    # Transform String type date to Factor type date
    start = date(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = date(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))

    # Make Datetime Index as minutes unit
    ix = pd.DatetimeIndex(start=start, end=end, freq='1min')

    # The number of total merged csv file
    data_len = len(files)

    #By n_jobs, split file list
    range_start = 0
    range_end = int(data_len / n_jobs)

    SplitByCount_files = []
    SplitByCount_tags = []

    output = []
    procs = []
    dfs = []

    # Split data list
    for j in range(n_jobs):
        if j == n_jobs-1:
            range_end = len(files)

        SplitByCount_files.append(files[range_start:range_end])
        SplitByCount_tags.append(tags[range_start:range_end])
        range_start = range_end
        range_end = range_end + int(data_len / n_jobs)

    # Make queue list to insert process result
    for i in range(n_jobs):
        output.append(Queue())

    # Make N process to execute task
    for i in range(n_jobs):
        procs.append(Process(target=getDataFrame, args=(SplitByCount_files[i], SplitByCount_tags[i], ix, output[i])))

    # Start process
    for p in procs:
        print('excute process')
        print(p)
        p.start()

    # 프로세스 처리가 끝날때 까지 대기, 프로세스 시작 순서대로 output을 queue에 삽입
    for output_df in output:
        print('get process')
        one_df = output_df.get()
        dfs.append(one_df)

    # queue 닫기
    for i in range(n_jobs):
        output[i].close

    # process 종료
    for p in procs:
        print('join process\n')
        p.join()

    # 분할 처리한 결과 병합
    final_df = concat_df(dfs)
    final_df = final_df.set_index(ix[:len(ix)-1])


    return final_df

def concat_df(dfs):
    df = pd.concat((df for df in dfs), axis=1)

    return df