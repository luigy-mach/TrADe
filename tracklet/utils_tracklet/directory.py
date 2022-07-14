
import os




def join_path(path1,path2):
    return os.path.join(path1,path2)
                    
def check_exist(path):
    """Return true if it exists a file or a directory"""
    return os.path.exists(path)


def check_dir_exist( main_directory=None, sub_directory=None):
    if main_directory is None: 
        main_directory = './'
    else:
        if(os.path.exists(main_directory) is False):
            print("Error path "+ main_directory + " -> was assigned ./")
            main_directory = './'
    if sub_directory is None:
        main_directory = os.path.join(main_directory,"trash")
        try:
            os.mkdir(main_directory)
            print("Directory " , main_directory ,  " Created ")
            return main_directory
        except FileExistsError:
            print("Directory " , main_directory ,  " already exists")
        return main_directory
    
    main_directory = os.path.join(main_directory, sub_directory)
    try:
        os.mkdir(main_directory)
        print("Directory " , main_directory ,  " Created ")
        return main_directory
    except FileExistsError:
        print("Directory " , main_directory ,  " already exists")
    return main_directory


# def create_dir(name_dir, show_msg=True):
#     try:
#         os.mkdir(name_dir)
#         if show_msg:
#             print("Directory " , name_dir ,  " Created ")
#         return True
#     except FileExistsError:
#         if show_msg:
#             print("Directory " , name_dir ,  " already exists")
#         return False
        
def create_dir(path):
    try:
        os.makedirs( path ,exist_ok=True)
        return
    except:
        print("{} already exist".format(path))
        return



def create_dir_test(path_results, prefix=None, dirtype=None):


    if prefix is None:
        prefix = 'test'
    version_pack = 0

    dir_vesion = '{}_v{}'
    path_pack  = os.path.join(path_results, dir_vesion.format(prefix, version_pack))

    while(os.path.exists(path_pack)):
        path_pack = os.path.join(path_results, dir_vesion.format(prefix, version_pack))
        version_pack+=1
    create_dir(path_pack)

    if dirtype=='crop':
        print(path_results)
        # ###################################################
        dir_gallery     = os.path.join(path_pack,'cropping')
        dir_log         = os.path.join(path_pack,'log')
        dir_parameters  = os.path.join(path_pack,'parameters_video')

        create_dir(dir_gallery)
        create_dir(dir_log)
        create_dir(dir_parameters)
        return dir_gallery, dir_log, dir_parameters

    return path_pack
