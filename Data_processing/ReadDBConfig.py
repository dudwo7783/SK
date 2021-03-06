import configparser
import pandas as pd

class ReadDBConfig:
    id = ''
    password = ''
    host = ''
    database = ''
    TableTitles = []
    DbTable = {}
    Count = 0

    new_date = ''
    new_tag_list = []
    cfg = None

    def __init__(self,path):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(path)

    def get_table(self, section):
        self.id = self.cfg.get('DB', 'id')
        self.password = self.cfg.get('DB', 'password')
        self.host= self.cfg.get('DB', 'host')
        self.database = self.cfg.get('DB', 'database')

        table_names = self.cfg.get('DB', 'tablelist')
        self.TableTitles = table_names.split(' ')
        self.Count = len(self.TableTitles)

        self.new_date = self.cfg.get('DATA', 'new_date')
        new_tag = self.cfg.get('DATA', 'new_tag')
        self.new_tag_list = new_tag.split(' ')

        for table in self.TableTitles:
            self.DbTable[table] = self.cfg.get(section, table).split(' ')
    '''
    def get_table_frame(self, RTDB, table_name):
        df = []
        table_elements = self.DbTable[table_name]

        for element in table_elements:
            a = RTDB[element]
            temp_df = a.reset_index()
            temp_df.columns= ['date','value']
            df.append(temp_df)

        final_df = pd.concat(df, keys=table_elements).reset_index(level = 0).reset_index(drop=True)
        final_df.columns = ['tag_code', 'time', 'value']
        print(final_df)

        return final_df
        '''

    def get_table_frame(self, RTDB, table_name):
        table_elements = self.DbTable[table_name]

        final_df = RTDB[table_elements].reset_index()

        final_df.columns = ['date'] + table_elements
        print(final_df)

        return final_df