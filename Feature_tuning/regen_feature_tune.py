import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from datetime import timedelta
from datetime import datetime as dt
from multiprocessing import Process, Queue
import numpy as np
import time as tm
from collections import OrderedDict
import os


# rds 경로
coke_path = r'.\RDS\Total_No2REF_Regenerator.rds'
FNAP_path = r'.\RDS\Predict Feed(NAPHTHENE) data_20170508_Final.rds'
FPAR_path = r'.\RDS\Predict Feed(PARAFFINE) data_20170508_Final.rds'
PPAR_path = r'.\RDS\Predict Product(PARAFFINE) data_20170508_Final.rds'
PARO_path = r'.\RDS\Predict Product(AROMATIC) data_20170508_Final_lm.rds'

# 최종 RTDB가 저장될 경로
RTDB_file_name = r'./regen_feature_data/RTDB.csv'

# 기준 시간
time_dic = OrderedDict({'start': 0, 'gas_2nd_time': 30, 'gas_1st_time': 150, 'coke_time': 180, 'backcoke_time' : -120, 'density_time' : -90,
            'R4_RIT_END': 240, 'R4_RIT_START': 2060, 'R3_RIT_END': 2120, 'R3_RIT_START': 3070,
            'R2_RIT_END': 3110, 'R2_RIT_START': 3680, 'R1_RIT_END': 3710, 'R1_RIT_START': 4160,
            })

# 주요 태그 설정
raw_value_dic = OrderedDict([
    ('max_2nd.Burning.Zone', 'start'),
    ('max_2nd.1dan', 'start'),
    ('max_2nd.2dan', 'start'),
    ('max_2nd.3dan', 'start'),
    ('max_2nd.4dan', 'start'),
    ('max_2nd.5dan', 'start'),
    ('Catalyst Circulation Rate', 'coke_time'),
    ('Catalyst Circulation LIFT VELOCITY E', 'coke_time'),
    ('Catalyst_V3_front_coke Content', 'coke_time'),
    ('Catalyst_V3_back_coke Content' , 'backcoke_time'),
    ('Catalyst_V3_front_cl Content' , 'coke_time'),
    ('Catalyst_V3_back_cl Content' , 'backcoke_time'),
    ('V-3 1ST BURN ZON E-1 INLET O2 CCT', 'gas_1st_time'),
    ('V-3 1ST BURN ZON E-1 INLET GAS', 'gas_1st_time'),
    ('V-3 1ST BURN ZON H-1 OUTLET', 'gas_1st_time'),
    ('Burning Gas C-2 OUT TO V-3', 'gas_2nd_time'),
    ('ATMOSPHERE Burning Gas TEMP', 'gas_2nd_time'),
    ('H-4 INLET TEMPERATURE (R-4 RIT)', 'R4_RIT_START'),
    ('H-3 OUT R-3 IN (R-3 RIT)', 'R3_RIT_START'),
    ('H-2 OUT R-2 IN (R-2 RIT)', 'R2_RIT_START'),
    ('H-1 OUT R-1 IN (R-1 RIT)', 'R1_RIT_START'),
    ('R-4 OUT E-1 IN (R-4 ROT)', 'R4_RIT_END'),
    ('R-3 OUT H-4 IN (R-3 ROT)', 'R3_RIT_END'),
    ('R-2 OUT H-3 IN (R-2 ROT)', 'R2_RIT_END'),
    ('R-1 OUT H-2 IN (R-1 ROT)', 'R1_RIT_END'),
    ('Burning Gas concentration' , 'density_time')
])

raw_value_keys = [
    'max_2nd.Burning.Zone',
    'max_2nd.Burning.1dan',
    'max_2nd.Burning.2dan',
    'max_2nd.Burning.3dan',
    'max_2nd.Burning.4dan',
    'max_2nd.Burning.5dan',
    'CC.Rate',
    'CC.VELOCITY',
    'CC.Frontcoke.content',
    'CC.Backcoke.content',
    'CC.Frontcl.content',
    'CC.Backcl.content',
    'gas.o2.pure.1st',
    'gas.flow.1st',
    'gas.temp.1st',
    'gas.flow.2nd',
    'gas.temp.2nd',
    'R4_RIT_value',
    'R3_RIT_value',
    'R2_RIT_value',
    'R1_RIT_value',
    'R4_ROT_value',
    'R3_ROT_value',
    'R2_ROT_value',
    'R1_ROT_value',
    'Burning Gas concentration'
]

# Reactor #1 ~ #4 Mean
mean_R1_to_4 = OrderedDict([
    ('R4_RIT_mean', 'H-4 INLET TEMPERATURE (R-4 RIT)'),
    ('R3_RIT_mean', 'H-3 OUT R-3 IN (R-3 RIT)'),
    ('R2_RIT_mean', 'H-2 OUT R-2 IN (R-2 RIT)'),
    ('R1_RIT_mean', 'H-1 OUT R-1 IN (R-1 RIT)')
])

# Reactor #1 ~ #4 Max
max_R1_to_4 = OrderedDict({
    'R4_RIT_max': 'H-4 INLET TEMPERATURE (R-4 RIT)',
    'R3_RIT_max': 'H-3 OUT R-3 IN (R-3 RIT)',
    'R2_RIT_max': 'H-2 OUT R-2 IN (R-2 RIT)',
    'R1_RIT_max': 'H-1 OUT R-1 IN (R-1 RIT)'
})

# Regen gas
gas_var = OrderedDict({
    'Re_Gas_H2_purity':['Recycle Gas H2 purity'],
    'Re_Gas_Flow' :['REC GAS TO E-1 Flow'],
    'Re_Gas_MOLAR_RATIO':['H2/HC MOLAR RATIO'],
    'C_1_DISCH':['C-1 DISCH'],
    'C_1_SUCTION' :['C-1 SUCTION']
})

# New wait
wait_var = OrderedDict({
    'R4_New_Wait': ['New_Wait'],
    'R3_New_Wait': ['New_Wait'],
    'R2_New_Wait': ['New_Wait'],
    'R1_New_Wait': ['New_Wait']
})

# Reactor #1 ~ #4 변환총량, 노출총량, Feed Total
FP_total_R1_to_4 = OrderedDict({
    'R4_RDT_PAR_FEED':
        [
            'H-4 INLET TEMPERATURE (R-4 RIT)',
            'R-4 OUT E-1 IN (R-4 ROT)',
            'hat_y_PAR'
        ],
    'R3_RDT_PAR_FEED':
        [
            'H-3 OUT R-3 IN (R-3 RIT)',
            'R-3 OUT H-4 IN (R-3 ROT)',
            'hat_y_PAR'
        ],
    'R2_RDT_PAR_FEED':
        [
            'H-2 OUT R-2 IN (R-2 RIT)',
            'R-2 OUT H-3 IN (R-2 ROT)',
            'hat_y_PAR'
        ],
    'R1_RDT_PAR_FEED':
        [
            'H-1 OUT R-1 IN (R-1 RIT)',
            'R-1 OUT H-2 IN (R-1 ROT)',
            'hat_y_PAR'
        ],
    'R4_RDT_T_ARO':
        [
            'H-4 INLET TEMPERATURE (R-4 RIT)',
            'R-4 OUT E-1 IN (R-4 ROT)',
            'Transter_ARO'
        ],
    'R3_RDT_T_ARO':
        [
            'H-3 OUT R-3 IN (R-3 RIT)',
            'R-3 OUT H-4 IN (R-3 ROT)',
            'Transter_ARO'
        ],
    'R2_RDT_T_ARO':
        [
            'H-2 OUT R-2 IN (R-2 RIT)',
            'R-2 OUT H-3 IN (R-2 ROT)',
            'Transter_ARO'
        ],
    'R1_RDT_T_ARO':
        [
            'H-1 OUT R-1 IN (R-1 RIT)',
            'R-1 OUT H-2 IN (R-1 ROT)',
            'Transter_ARO'
        ],
    'R4_RDT_T_PAR':
        [
            'H-4 INLET TEMPERATURE (R-4 RIT)',
            'R-4 OUT E-1 IN (R-4 ROT)',
            'Transter_PAR'
        ],
    'R3_RDT_T_PAR':
        [
            'H-3 OUT R-3 IN (R-3 RIT)',
            'R-3 OUT H-4 IN (R-3 ROT)',
            'Transter_PAR'
        ],
    'R2_RDT_T_PAR':
        [
            'H-2 OUT R-2 IN (R-2 RIT)',
            'R-2 OUT H-3 IN (R-2 ROT)',
            'Transter_PAR'
        ],
    'R1_RDT_T_PAR':
        [
            'H-1 OUT R-1 IN (R-1 RIT)',
            'R-1 OUT H-2 IN (R-1 ROT)',
            'Transter_PAR'
        ],
    'R4_RDT_Total_Feed':
        [
            'H-4 INLET TEMPERATURE (R-4 RIT)',
            'R-4 OUT E-1 IN (R-4 ROT)',
            '340 FEED TOTAL Flow'
        ],
    'R3_RDT_Total_Feed':
        [
            'H-3 OUT R-3 IN (R-3 RIT)',
            'R-3 OUT H-4 IN (R-3 ROT)',
            '340 FEED TOTAL Flow'
        ],
    'R2_RDT_Total_Feed':
        [
            'H-2 OUT R-2 IN (R-2 RIT)',
            'R-2 OUT H-3 IN (R-2 ROT)',
            '340 FEED TOTAL Flow'
        ],
    'R1_RDT_Total_Feed':
        [
            'H-1 OUT R-1 IN (R-1 RIT)',
            'R-1 OUT H-2 IN (R-1 ROT)',
            '340 FEED TOTAL Flow'
        ]
})

# 병렬 처리 target
target_var = [
        [FP_total_R1_to_4, 'cumulative', True],
        [mean_R1_to_4, 'mean', True],
        [max_R1_to_4, 'max', True],
        [gas_var, 'gas', False],
        [wait_var, 'cumulative', True]
    ]


# 병렬 처리 프로세스
# base_df :
# rtdb_df : coke data rtdb
def merge_parallel_data(feature, base_df, rtdb_df):
    start_time = tm.time()
    print('parallel computing start')

    output = []
    procs = []
    dfs = []
    true_count = 0

    var_N = len(feature)

    # 병렬 처리 해야하는 딕셔너리 개수 카운트(#R1 ~ #R4 별로 처리해야하는 태그)
    for var in feature:
        if True in var:
            true_count = true_count +1

    # 병렬 처리 딕셔너리 * 4(R1~R4) + 개별 태그
    total_output_count = (true_count * 4) + (var_N - true_count)

    # 총 처리해야할 프로세스의 결과를 저장할 큐 리스트 생성
    for i in range(total_output_count):
        output.append(Queue())

    output_step = 0

    for i in range(var_N):
        feature_dict = feature[i][0]    # max_R1_to_4, mean_R1_to_4,gas_var,wait_var,FP_total_R1_to_4 중 선택
        what = feature[i][1]            # 무엇을 처리할지 선택(평균, 최댓값, 총량..)
        isStep = feature[i][2]          # regen 단계가 존재하는지 boolean

        # regen 단계가 있는 것들은 #R1~ #R4 별로 프로세스 분할
        if isStep == True:
            procs.append(Process(target=creat_feature, args=(feature_dict, 'R1', what, base_df, rtdb_df, output[output_step])));output_step = output_step+1
            procs.append(Process(target=creat_feature, args=(feature_dict, 'R2', what, base_df, rtdb_df, output[output_step])));output_step = output_step+1
            procs.append(Process(target=creat_feature, args=(feature_dict, 'R3', what, base_df, rtdb_df, output[output_step])));output_step = output_step+1
            procs.append(Process(target=creat_feature, args=(feature_dict, 'R4', what, base_df, rtdb_df, output[output_step])));output_step = output_step+1

        # 단계가 없는것들(gas_var)
        else:
            procs.append(Process(target=creat_feature, args=(feature_dict, 'X', what, base_df, rtdb_df, output[output_step])));output_step = output_step+1

    # 프로세스 시작
    for p in procs:
        print('excute process')
        print(p)
        p.start()

    # 프로세스 처리가 끝날때 까지 대기, 프로세스 시작 순서대로 output을 queue에 삽입
    for output_df in output:
        print('get process')
        one_df = output_df.get()
        dfs.append(one_df)

    print('exit get process')

    # queue 닫기
    for i in range(total_output_count):
        output[i].close

    # process 종료
    for p in procs:
        print('join process\n')
        p.join()

    # 최종적으로 얻은 튜닝된 태그들 병합
    final_df = concat_df(dfs)

    final_df = final_df.set_index(base_df['start'])
    del final_df.index.name

    #final_df.columns = feature.keys()

    print("parallel computing end--- %s seconds ---" % (tm.time() - start_time))

    return final_df


def concat_df(dfs):
    df = pd.concat((df for df in dfs), axis=1)

    return df

# 변수 생성 함수
# feature : [max_R1_to_4, mean_R1_to_4, gas_var, wait_var, FP_total_R1_to_4] 중 1
# step : [R1, R2, R3, R4] 중 1
# what : [max, mean, gas, new wait, 총량] 중 1
# base_df : time lag information
# rtdb_df : tag data
# queue : 결과가 저장될 memory
def creat_feature(feature, step, what, base_df, rtdb_df, queue):

    start_time = tm.time()
    feature_df = pd.DataFrame()

    if 'max' in what:
        # name  : 처리 후 저장할  column 이름(step이 있는 딕셔너리들의 이름은 R1_, R2_, ..로 시작)
        # tag   : 처리할 때 필요한 tag 정보

        for name, tag in feature.items():
            if name[0:2] in step:
                # name[0:2] : R1, R2.. 를 의미, 즉 해당되는 단계에 대해서만 처리(개별 프로세스)

                print('-------- %s : %s -------' % (name, tag))
                # Make R#'s max column
                temp = max_feature(tag, step, base_df, rtdb_df)

                feature_df = pd.concat([feature_df, temp.rename(name)], axis=1)

                print("---%s - %s - %s seconds ---" % (name, tag, (tm.time() - start_time)))
            else:
                pass


    elif 'mean' in what:
        for name, tag in feature.items():

            if name[0:2] in step:
                print('-------- %s : %s -------' % (name, tag))
                # Make R#'s mean column
                temp = mean_feature(tag, step, base_df, rtdb_df)

                feature_df = pd.concat([feature_df, temp.rename(name)], axis=1)

                print("---%s - %s - %s seconds ---" % (name, tag, (tm.time() - start_time)))
            else:
                pass


    elif 'cumulative' in what:
        for name, tag in feature.items():

            if name[0:2] in step:
                print('-------- %s : %s -------' % (name, tag))
                # Make R#'s cumulative column
                temp = cumulative_feature(tag, step, base_df, rtdb_df)

                feature_df = pd.concat([feature_df, temp.rename(name)], axis=1)

                print("---%s - %s - %s seconds ---" %(name, tag, (tm.time() - start_time)))
            else:
                pass

    # gas는 regen의 단계별 처리가 필요없으므로 step 인자 제외
    elif 'gas' in what:
        for name, tag in feature.items():
            print('-------- %s : %s -------' % (name, tag))
            # Make gas cumulative column
            temp = gas_feature(tag, 'gas_2nd_time', base_df, rtdb_df)

            feature_df = pd.concat([feature_df, temp.rename(name)], axis=1)

            print("---%s - %s - %s seconds ---" % (name, tag, (tm.time() - start_time)))

    else:
        pass

    queue.put(feature_df)
    #return feature_df


def max_feature(tag, step, base_df, rtdb_df):
    start = step + '_RIT_START'
    end = step + '_RIT_END'

    # RTDB의 해당 태그의(기준 시작 날짜 ~ 기준 끝 날짜) 최댓값 찾기
    # 기준 시작 날짜 ~ 기준 끝 날짜의 시간 단위만큼 수행
    max_df = base_df[[start, end]].apply(
        lambda x: np.max(rtdb_df[x[start]:x[end]][tag]), axis=1
    )

    return max_df


def mean_feature(tag, step, base_df, rtdb_df):
    start = step + '_RIT_START'
    end = step + '_RIT_END'

    # RTDB의 해당 태그의(기준 시작 날짜 ~ 기준 끝 날짜) 평균값 찾기
    # 기준 시작 날짜 ~ 기준 끝 날짜의 시간 단위만큼 수행
    mean_df = base_df[[start, end]].apply(
        lambda x: np.mean(rtdb_df[x[start]:x[end]][tag]), axis=1
    )

    return mean_df


def gas_feature(tag, start, base_df, rtdb_df):

    # RTDB에서 기준날짜의 해당 태그 값 찾기
    gas_df = base_df[[start]].apply(
        lambda x: rtdb_df.loc[x[start], tag[0]], axis=1
    )

    return gas_df


def cumulative_feature(tag, step, base_df, rtdb_df):
    start = step + '_RIT_START'
    end = step + '_RIT_END'

    N_tag = len(tag)

    # 필요한 tag가 3개이면 총량 계산
    # ex : [RIT, ROT, hat_y_PAR]
    if N_tag == 3:
        # 기준 날짜 기간의 {(RIT-ROT)*추정값} 의 총합
        cumulative_df = base_df[[start, end]].apply(
            lambda x: np.sum(
                (rtdb_df[x[start]:x[end]][tag[0]]
                 - rtdb_df[x[start]:x[end]][tag[1]])
                * rtdb_df[x[start]:x[end]][tag[2]]
            ), axis=1
        )

        return cumulative_df

    # 필요한 tag가 1개이면 new wait 찾기
    elif N_tag == 1:
        cumulative_df = base_df[[start]].apply(
            lambda x: rtdb_df.loc[x[start], tag[0]], axis=1
        )

        return cumulative_df

    else:
        pass


if __name__ == "__main__":

    # rds 파일을 읽기 위한 세팅
    pandas2ri.activate()
    readRDS = robjects.r['readRDS']

    # Read rds file
    coke_data = pandas2ri.ri2py(readRDS(coke_path))  # 2016-12-27 ~
    Feed_NAP = pandas2ri.ri2py(readRDS(FNAP_path))  # 2014-01-01 ~
    Feed_PAR = pandas2ri.ri2py(readRDS(FPAR_path))  # 2014-01-01 ~
    Product_PAR = pandas2ri.ri2py(readRDS(PPAR_path))  # 2017-01-01 ~
    Product_ARO = pandas2ri.ri2py(readRDS(PARO_path))  # 2017-01-01 ~

    # Change second to datetime
    coke_data['time'] = coke_data['time'].apply(lambda x: dt.fromtimestamp(x))
    Feed_NAP['time'] = Feed_NAP['time'].apply(lambda x: dt.fromtimestamp(x))
    Feed_PAR['time'] = Feed_PAR['time'].apply(lambda x: dt.fromtimestamp(x))
    Product_PAR['time'] = Product_PAR['time'].apply(lambda x: dt.fromtimestamp(x))
    Product_ARO['time'] = Product_ARO['time'].apply(lambda x: dt.fromtimestamp(x))

    tag_list = [coke_data, Feed_NAP, Feed_PAR, Product_PAR, Product_ARO]

    # Set index to datetime
    for i in range(0, len(tag_list)):
        tag_list[i] = tag_list[i].set_index('time')

    Coke_Data_FPRC = tag_list[0]

    # Merge all dataframe
    for i in range(1, len(tag_list)):
        Coke_Data_FPRC = pd.concat([Coke_Data_FPRC, tag_list[i]], axis=1, join='inner')

    # Add Feed prediction (PAR, NAP)
    Coke_Data_FPRC = Coke_Data_FPRC.assign(
        hat_y_ARO=pd.Series(100 - Coke_Data_FPRC['hat_y_PAR'] - Coke_Data_FPRC['hat_y_NAP'])
    )
    # Add Product prediction (ARO)
    Coke_Data_FPRC = Coke_Data_FPRC.assign(
        Transter_ARO=pd.Series(np.absolute(Coke_Data_FPRC['hat_Product_y_ARO'] - Coke_Data_FPRC['hat_y_ARO']))
    )
    # Add Product prediction (PAR)
    Coke_Data_FPRC = Coke_Data_FPRC.assign(
        Transter_PAR=pd.Series(np.absolute(Coke_Data_FPRC['hat_Product_y_PAR'] - Coke_Data_FPRC['hat_y_PAR']))
    )
    # Add New wait
    Coke_Data_FPRC = Coke_Data_FPRC.assign(
        New_Wait=pd.Series(0.125 * Coke_Data_FPRC['H-1 OUT R-1 IN (R-1 RIT)'] +
                           0.157 * Coke_Data_FPRC['H-2 OUT R-2 IN (R-2 RIT)'] +
                           0.256 * Coke_Data_FPRC['H-3 OUT R-3 IN (R-3 RIT)'] +
                           0.463 * Coke_Data_FPRC['H-4 INLET TEMPERATURE (R-4 RIT)'])
    )

    # Set time interval
    base_FRC = Coke_Data_FPRC.loc['2017-01-04':'2017-05-09']
    base_FRC = pd.DataFrame({'start': base_FRC.index.tolist()})

    base_FRC_tag = list(time_dic.keys()) + list(raw_value_dic.keys())

    # Calculate Time(start, coke, RIT START, RIT END...)
    base_FRC_date = pd.concat(((base_FRC['start'] - timedelta(minutes=time)) for name, time in time_dic.items()),
                              axis=1, keys=time_dic.keys())
    # Concat Primary dataframe

    base_FRC_value = pd.concat(((Coke_Data_FPRC.loc[base_FRC_date[time], name].reset_index(drop=True)) for name, time in
                                raw_value_dic.items()),
                               axis=1, keys=raw_value_keys)
    # Merge Date's data frame and Primary's dataframe
    base_FRC = pd.concat([base_FRC_date, base_FRC_value], axis=1)

    base_FRC_value = base_FRC_value.set_index(base_FRC_date['start'])

    base_FRC.to_csv(r'./regen_feature_data/date.csv')

    # If RTDB file exist, remove original file
    if os.path.exists(RTDB_file_name):
        os.remove(RTDB_file_name)

    # Preprocessing process start
    RTDB = merge_parallel_data(target_var, base_FRC, Coke_Data_FPRC)
    RTDB.to_csv(r'./regen_feature_data/tune_data.csv')
    # Merge Created feature dataframe, primary dataframe and date dataframe
    total_RTDB = concat_df([RTDB, base_FRC_value])

    total_RTDB.index = base_FRC['start']
    # Save RTDB to csv file
    total_RTDB.to_csv(RTDB_file_name)

    print(total_RTDB)
