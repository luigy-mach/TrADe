import os
import pandas as pd
import numpy as np

def save_data( col_names, dict_result, savePath):

    dataframe = pd.DataFrame(columns=col_names)
    values    = list()
    for i in col_names:
        values.append(dict_result[i])
    dataframe.loc[len(dataframe)] = values
    if os.path.exists(savePath):
        dataframe.to_csv(savePath, mode='a', index=False, header=False)
    else:
        dataframe.to_csv(savePath, index=False, header=True)


def find_files(path_main, pattern, type='separate'):

    list_return = list()
    for dirpath, dirs, files in os.walk(path_main):
        for fname in fnmatch.filter(files, pattern):
            list_return.append((dirpath,fname))

    if type == 'separate':
        list_return = np.asarray(list_return)
        df          = pd.DataFrame(list_return, columns = ['path','file'])
        df          = df.sort_values(by=['path'], ascending=True)
        return df.to_numpy()

    if type == 'absolute':
        new_list = list()
        for i,j in list_return:
            new_list.append(os.path.join(i,j))
        new_list = sorted(new_list )
        return np.asarray(new_list)
    else:
        print('error, you need choise type: [separate,absolute]')
        return None


def check_id_into_gts(gts_dfPath, id):
    if isinstance(id,str):
        id = int(id)
    assert id>=0, 'error, id is negative'
    df = pd.read_csv(gts_dfPath)
    ids = df['id'].to_numpy()
    ids = np.unique(ids)
    qID = np.where(ids==id)
    if qID[0].size > 0:
        return True
    else:
        return False