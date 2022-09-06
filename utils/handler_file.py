import os
import shutil

from .myglobal import key_words as dictkeywords


import json
import datetime
import re

import numpy  as np
import pandas as pd
import fnmatch



def find_files(path_main, pattern, type='separate'):
    list_return = list()
    for dirpath, dirs, files in os.walk(path_main):
        for fname in fnmatch.filter(files, pattern):
            list_return.append((dirpath,fname))

    if len(list_return)==0:
        print("error, not found files.")
        return None

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

def find_files_yield(path_main, pattern, type='separate'):
    if type == 'separate':
        list_return = list()
        for dirpath, dirs, files in os.walk(path_main):
            for fname in fnmatch.filter(files, pattern):
                yield (dirpath,fname)
    if type == 'absolute':
        list_return = list()
        for dirpath, dirs, files in os.walk(path_main):
            for fname in fnmatch.filter(files, pattern):
                yield os.path.join(dirpath,fname)
    else:
        return False


def find_dirs(path_main, pattern, type='separate'):
    list_return = list()
    for dirpath, dirs, files in os.walk(path_main):
        for dname in fnmatch.filter(dirs, pattern):
            list_return.append((dirpath,dname))
    if type == 'separate':
        list_return = np.asarray(list_return)
        df          = pd.DataFrame(list_return, columns = ['path','dir'])
        # df            = df.sort_values(by=['path'], ascending=True)
        return df.to_numpy()

    if type == 'absolute':
        new_list = list()
        for i,j in list_return:
            new_list.append(os.path.join(i,j))
        # new_list = sorted(new_list )
        return np.asarray(new_list)
    else:
        print('error, you need choise type: [separate,absolute]')
        return None



def find_dirs_yield(path_main, pattern, type='separate'):
    if type == 'separate':
        list_return = list()
        for dirpath, dirs, files in os.walk(path_main):
            for dname in fnmatch.filter(dirs, pattern):
                yield (dirpath,dname)
    if type == 'absolute':
        list_return = list()
        for dirpath, dirs, files in os.walk(path_main):
            for dname in fnmatch.filter(dirs, pattern):
                yield os.path.join(dirpath,dname)
    else:
        return False

def linspace(start, end, step=1.):
    assert start<end, "start need greater than end"
#     epsilon = 0.000001
#     result = np.arange(start,end+epsilon,step)
    result = np.arange(start,end,step)
    result =  np.around(result, decimals=3)
    result = list(result)
    if result[-1]!=end:
        result.append(end)     
    return result


def read_files_from_dir(path, extension='.png'):
    assert os.path.exists(path), 'directory {} doesnt exist'.format(path)
    files   = list(os.listdir(path))
    pattern = re.compile(r'[\d]+_inliers_[\d]+_frame')
    files   = list(filter(pattern.search, files))
    files   = sorted([ i for i in files if i.endswith(extension)])
    return files



def save_set_time(dict_reid, fps, name_save=None, path_save=None):

    times = list()
    for k, v in list(dict_reid.items()):
        current_frame        = int(v['frame'][0])
        time                 = str(datetime.timedelta(seconds = current_frame/fps))
        dict_reid[k]['time'] = [time]

    if path_save:
        if not name_save:
            name_save = 'test.json'
        path = os.path.join(path_save, name_save)
        
        with open(path,'w') as file:
            json.dump(dict_reid, file, indent=4)



def split_text(text, separator='.'):    
    return text.split(separator)


def split_fpath(text):
    dict_result = dict()
    list_split  = split_text(split_text(text, '.')[0], '_')
    for i, item in enumerate(list_split):
        num_step = dictkeywords.get(item)
        if num_step is not None:
            dict_result[item] = list_split[i+1:i+1+num_step]
    return dict_result



def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        # print('{} created successfully'.format(path))
        return path
    except:
        print('{} cant be created')
    return None


def remove_folder_contents(path):
    try:
        shutil.rmtree(path, ignore_errors=True)
        create_dir(path)
        return path
    except:
        print('err, remove_folder_contents() - function')
        return None


def generate_new_pack(path, name=None, version=None, posfix=None):
    _name    = name
    _version = version
    if _name is None:
        _name = 'pack'
    if _version is None:
        _version = 0

    if posfix is not None:
        name_dir = '{}_v{}_{}'.format(_name, _version, posfix)
        while(os.path.exists(os.path.join(path,name_dir))):
            _version +=1
            name_dir = '{}_v{}_{}'.format(_name, _version, posfix)
    else:
        name_dir = '{}_v{}'.format(_name, _version)
        while(os.path.exists(os.path.join(path,name_dir))):
            _version +=1
            name_dir = '{}_v{}'.format(_name, _version)


    path_save    = create_dir(os.path.join(path,name_dir))

    path_gallery = os.path.join(path_save, 'gallery_inliers')
    path_reid    = os.path.join(path_save, 'out_reid')
    path_video   = os.path.join(path_save, 'out_video')

    create_dir(path_gallery)
    create_dir(path_reid)
    create_dir(path_video)

    return path_gallery, path_reid, path_video



def check_exists(*folders):
    try:
        for i in folders:
            assert os.path.exists(i), '{} doesnt exists'.format(º)
        return True
    except:
        print('error check_exists()')
        return False





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


def find_dirs(path_main, pattern):

    list_return = list()
    for dirpath, dirs, files in os.walk(path_main):
        for dname in fnmatch.filter(dirs, pattern):
            list_return.append((dirpath,dname))

    list_return = np.asarray(list_return)
    # return list_return
    df = pd.DataFrame(list_return, columns = ['path','dir'])
    df = df.sort_values(by=['path'], ascending=True)
    return df.to_numpy()


def copy_files_from_array(array_src, pathDst, extension='*.png'):
    try:
        for i, absPath in enumerate(array_src):
            pathFile, nameFile = os.path.split(absPath)
            srcPath            = os.path.join(pathFile, nameFile)
            nameFile           = "{}_{}".format(i, nameFile)
            dstPath            = os.path.join(pathDst, nameFile)
            shutil.copyfile(srcPath, dstPath)
        return True

    except:
        return print("Error, Destination is a directory. (def copy_files_from_array)")
