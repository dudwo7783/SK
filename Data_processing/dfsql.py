# Pandas Dataframe sql 저장 모듈

import pymysql
import pymssql
from sqlalchemy import create_engine


def db_session(id, password, host, database):
    engine = create_engine('mysql+pymysql://'+id+':'+password+'@'+host+'/'+database+'?charset=utf8',
                           convert_unicode=True)
    return engine

'''
def db_session(id, password, host, database):
    engine = create_engine('mssql+pymssql://'+id+':'+password+'@'+host+'/'+database+'?charset=utf8',
                           convert_unicode=True)
    return engine
'''
