
import os
import shutil

def create_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except :
        print('error, create_dir(path)')
        return None


def remove_directory(dir_path):
    try:
        assert os.path.exists(dir_path)
        if os.path.isfile(dir_path):
            shutil.rmtree(dir_path)
            return True
        else:
            shutil.rmtree(dir_path)
            return True

    except:
        print('Error: {}'.format(dir_path))
        return False