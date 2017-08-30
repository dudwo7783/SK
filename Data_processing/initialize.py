#-*- codig: utf-8 -*-

# Made by Kim Youngjae
# 17-08-16
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os.path import join, exists
from os import mkdir,remove
import time as tm
import pandas as pd
from sqlalchemy import types as sqltypes

import data_processing as dp
import check_file_name as cfn
from ReadDBConfig import ReadDBConfig
import dfsql

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data_dir',
      type=str,
      default=r'.\raw_data',
      help='Raw data path.\ndefault : ./raw_data'
    )
    parser.add_argument(
      '--save_dir',
      type=str,
      default=r'.\RTDB',
      help='RTDB path to save RTDB.csv\ndefault : ./RTDB_dir'
    )
    parser.add_argument(
      '--start_date',
      type=str,
      default='2017-04-20',
      help='Start date (2017-01-01)'
    )
    parser.add_argument(
      '--end_date',
      type=str,
      default='2017-05-9',
      help='End date (2017-05-10)'
    )
    parser.add_argument(
        '--rm_old_merge_file',
        type=str,
        default='No',
        help='Delete old merged files (Yes or No)'
    )
    parser.add_argument(
      '--split_data',
      type=str,

      default=None,
      nargs='+',
      help='Divided data by month or year\nAdd multiple file using white space.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    data_dir = FLAGS.data_dir
    save_dir = FLAGS.save_dir
    start_date = FLAGS.start_date
    end_date = FLAGS.end_date

    if FLAGS.split_data == None:
        split_files = cfn.check_filename(data_dir)
    else:
        split_files = FLAGS.split_data

#####################################################################################################
    RTDB_file_name = join(save_dir, "RTDB.csv")

    config = ReadDBConfig(r'.\DB.conf')
    config.get_table('DB_TABLE')

    id = config.id
    password = config.password
    host = config.host
    database = config.database

    new_date = config.new_date
    new_taglist = config.new_tag_list


    db = dfsql.db_session(id, password, host, database)
#####################################################################################################


#####################################################################################################
    '''
    RTDB = pd.read_csv('.\RTDB\RTDB.csv', index_col=0)

    config = ReadDBConfig(r'.\DB.conf')
    config.get_table('DB_TABLE')

    DbTables = {}

    for i in range(config.Count):
        table_name = config.TableTitles[i]
        table_element = config.DbTable[table_name]
        DbTables[table_name] = config.get_table_frame(RTDB, table_name)
        '''

#####################################################################################################

# 기존 RTDB 삭제
# 없을 시 폴더 생성
#####################################################################################################

    if not exists(save_dir):
        print("Make %s directory......" %save_dir)
        mkdir(save_dir)
    else:
        if exists(RTDB_file_name):
            print("Remove old RTDB file.....")
            remove(RTDB_file_name)

#####################################################################################################

# 기존 병합된 csv 파일 삭제
#####################################################################################################
    if "Yes" in FLAGS.rm_old_merge_file:
        for file in split_files:
          if exists(join(data_dir, file + ".csv")):
              print("Remove old merged file : %s....." %file)
              remove(join(data_dir, file + ".csv"))
#####################################################################################################

# RTDB 생성
#####################################################################################################
    start_time = tm.time()

    print("Make RTDB.....")

    files, tags = dp.getFileList(data_dir, split_files)
    RTDB = dp.parallel_processing(files, tags, start_date, end_date)
    RTDB = dp.DataToNa(new_date, RTDB, new_taglist)
    RTDB.to_csv(RTDB_file_name)

    print("Complete making RTDB\n  --- %s seconds ---" % (tm.time() - start_time))

    start_time = tm.time()
    print("Make Database and Insert data.....")

#####################################################################################################

# DB용 dataframe 생성
#####################################################################################################
    RTDB = pd.read_csv('.\RTDB\RTDB.csv', index_col=0)

    DbTables = {}

    for i in range(config.Count):
        table_name = config.TableTitles[i]
        DbTables[table_name] = config.get_table_frame(RTDB, table_name)
        DbTables[table_name].to_sql(name=table_name, con=db, if_exists='replace', dtype={'date': sqltypes.DateTime})
        #DbTables[table_name].to_sql(name=table_name, con=db, if_exists='replace',dtype={'tag_code' : sqltypes.TEXT, 'time' : sqltypes.DateTime, 'value' : sqltypes.Float})
#####################################################################################################
    print("Complete Making Database Table\n--- %s seconds ---" % (tm.time() - start_time))
