import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from multiprocessing.dummy import Process, Event, Queue
# from multiprocessing import Pool as Pool
import time

from ..utils_tracklet.directory import create_dir

global g_id
g_id = np.iinfo(np.int).min

def set_id_global(_g_id):
    global g_id
    g_id = _g_id    

# def get_id_global(step_grow=1, g_id=0):
def get_id_global(step_grow=1):
    global g_id
    tmp = g_id
    g_id+=step_grow
    return tmp


def random_crop(frame, crop_w, crop_h):
    x_max = frame.shape[1] - crop_w
    y_max = frame.shape[0] - crop_h
    x = np.random.randint(0,x_max)
    y = np.random.randint(0,y_max)
#     crop = frame[y:y+crop_h, x:x+crop_w]
#     return crop
    return [x, y, x+crop_w, y+crop_h]

def generate_crops(frame, num_crop, dim):
    mydict = dict()
    global g_id
    for i in range(num_crop):
        crop = random_crop(frame,dim[0],dim[1])
        mydict[int(i+g_id)]=crop
#         mydict[i]=crop
    return mydict



def save_tracklet(mytracklet, path_main, name_video=None, separate=True):
# def save_tracklet(mytracklet, path_main, separate=True):
    if not isinstance(mytracklet, list): return False
    
    if name_video is not None:
        path_main = os.path.join(path_main, name_video)
        create_dir(path_main)
    # else :
    #     name_video = 'name_assing'
    #     path_main = os.path.join(path_main, name_video)
    #     create_dir(path_main)

    for track in mytracklet:
        _id_frame   = track[0]
        _crop       = track[1]
        _id_bbox    = track[2]
        x0,y0,x1,y1 = track[3]
        _name       = "frame_{:08d}_id_{:08d}_bbox_{:04d}_{:04d}_{:04d}_{:04d}.png"
        # _name     = "frame_{}_id_{}_bbox_{}_{}_{}_{}.png"
        _name       = _name.format(int(_id_frame), int(_id_bbox), int(x0),int(y0),int(x1),int(y1))
        
        if separate:
            _path = os.path.join(path_main, '{0:08d}'.format(int(_id_bbox)) )
            create_dir(_path)
        else:
            _path = os.path.join(path_main)
            create_dir(_path)
        _path = os.path.join(_path, _name)
        cv2.imwrite(_path, _crop)
    return

def loop_queue_save(global_queue, event, path_main, separate=True, flag_finish='DONE',name_video=None):
# def loop_queue_save(globl_queue, event, path_main, separate=True, flag_finish='DONE'):
    event.clear()
    while True:
        if not global_queue.empty():
            _tracklet = global_queue.get()
            if _tracklet == flag_finish:
                event.set()
                break
            save_tracklet(_tracklet, path_main, name_video, separate)
            
        if event.is_set():
            print("---> termine proceso sobre queue <----")
            break
        
def add_tracklet_queue(global_queue, tracklet):
    if not len(tracklet)>0:
        print("error: add_tracklet_queue")
        return False
    global_queue.put(tracklet)
    return True

