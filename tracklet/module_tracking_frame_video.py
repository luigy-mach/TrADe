

import glob
import numpy as np
import sys
import os.path
import os
import random

from PIL import Image

import cv2
import time
import json 


# basedir = os.path.dirname(__file__)
# sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

from object_detector.interface_yolov3_tf import obj_detection
from multitracker.re3.tracker.re3_tracker import  Re3Tracker

from library.class_tracklet import *
from library.parallel_save import *

import multiprocessing.dummy as mp

def is_into(min, value, max):
    return min<=value and value<=max

def box1_into_box2(boxA, boxB):
    xA0 = int(boxA[0])
    yA0 = int(boxA[1])
    xA1 = int(boxA[2])
    yA1 = int(boxA[3])
    xB0 = int(boxB[0])
    yB0 = int(boxB[1])
    xB1 = int(boxB[2])
    yB1 = int(boxB[3])
    #A into B
    if is_into(xB0,xA0,xB1) and is_into(xB0,xA1,xB1):
        if is_into(yB0,yA0,yB1) and is_into(yB0,yA1,yB1):
            return True
    return False


# bb_IOU (Intersection Over Union) output values between 0 and 1
def bb_IOU(boxA, boxB):
    if box1_into_box2(boxA,boxB) or box1_into_box2(boxB,boxA):
        return 1
    xA = max(int(boxA[0]), int(boxB[0]))
    yA = max(int(boxA[1]), int(boxB[1]))
    xB = min(int(boxA[2]), int(boxB[2]))
    yB = min(int(boxA[3]), int(boxB[3]))
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (int(boxA[2]) - int(boxA[0]) + 1) * (int(boxA[3]) - int(boxA[1]) + 1)
    boxBArea = (int(boxB[2]) - int(boxB[0]) + 1) * (int(boxB[3]) - int(boxB[1]) + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou



# find_biggest_id find the biggest ID
def max_id(initBoundingBox):
    tmp = int(-999)
    for u_id in initBoundingBox.keys():
        if int(u_id) > int(tmp):
            tmp = u_id
    return int(tmp)


# optimizar!!!!
def merge_initBoundingBox_v3(bb_src, bb_new_objdect, threshold_merge_bb=0.40):
    # print("////////////////////////////////////////////////////,//////////////")
    # print("merge_bb_src//////////////////////////////////////////////////////")
    # print(bb_src)
    # print("merge_bb_new_objdect//////////////////////////////////////////////")
    # print(bb_new_objdect)
    # print("//////////////////////////////////////////////////////////////////")
 

    dict_result = bb_src.copy()
    for key_i, value_i in bb_new_objdect.items():
        max_roi_value      = -1
        max_roi_key        = None
        dic_scores         = dict()
        list_ids_updates   = list()

        for key_j, value_j in bb_src.items():
            score = bb_IOU(value_i, value_j)
            if score > threshold_merge_bb:
                if max_roi_value <= score:
                    max_roi_value = score
                    max_roi_key = key_j
                dic_scores[key_j]=score
            else: 
                dic_scores[key_j]=0
        
        if max_roi_value > threshold_merge_bb:
            if max_roi_key in dic_scores:
                del dic_scores[max_roi_key]
                dict_result[max_roi_key] = value_i
                list_ids_updates.append(max_roi_key)
            for key, value in dic_scores.items(): # cuando hay una interseccion
                if value > threshold_merge_bb:
                    if key in dict_result: 
                        del dict_result[key]
                    else:
                        print("    no lo encontre")
        else:
            global global_id
            maxid = max_id(bb_src)
            if maxid >= global_id:
                global_id = maxid    
            global_id+=1 #update id++
            new_id = str(global_id)
            dict_result.update({new_id:value_i})
            list_ids_updates.append(max_roi_key)

    list_ids = [key for key in dict_result.keys() ]
    return list_ids, dict_result, list_ids_updates



def setbox2str(bbox):
    return str(int(bbox[1]))+'_'+str(int(bbox[3]))+'_'+str(int(bbox[0]))+'_'+str(int(bbox[2]))
    # return str(int(bbox[1]))+','+str(int(bbox[3]))+','+str(int(bbox[0]))+','+str(int(bbox[2]))


def join_path(path1,path2):
    return os.path.join(path1,path2)
                    
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


def setup_window(name_video, name_window, show_video=True):
    if show_video:
        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name_window, 640,480)
    cap = cv2.VideoCapture(name_video)
    # num_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.set(cv2.CAP_PROP_FPS, num_fps-10)
    if (cap.isOpened()==False): 
        print("Error opening video stream or file")
        exit()
    return cap
    

def get_setupcolor(key, lenght):
    # tam = len(test_output)
    tam = lenght
    return cv2.cvtColor(np.uint8([[[int(key) * 255 / tam, 128, 200]]]), cv2.COLOR_HSV2RGB).squeeze().tolist()
                    

def draw_bbox(frame, key, test_output, bbox):
    try:
        color = get_setupcolor(key,len(test_output))
        cv2.rectangle(  frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color, 10)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        texScale   = 3
        thickness  = 10
        # color_text = (200,0,0)
        color_text = color
        cv2.putText(frame,str(key),(int(bbox[0]),int(bbox[1])), font, texScale, color_text, thickness, cv2.LINE_AA)
        return True
    except:
        print("Error: draw_bbox(): ")
        return False

def create_dir(name_dir, show_msg=True):
    try:
        os.mkdir(name_dir)
        if show_msg:
            print("Directory " , name_dir ,  " Created ")
        return True
    except FileExistsError:
        if show_msg:
            print("Directory " , name_dir ,  " already exists")
        return False


def create_path_crop(dir_gallery, key, save_separate=False):
    base_path =''
    if save_separate:
        base_path = join_path(dir_gallery, str(key))
        create_dir(base_path, show_msg=False)
    else:
        base_path = dir_gallery
    return base_path

def show_window(name_video, frame, show_video=True, pause_video=False):
    if show_video:
        cv2.imshow(name_video, frame)
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        if pause_video and cv2.waitKey(10) & 0xFF == ord('p'):
            while True:
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break  
    return True

def crop_frame(frame, bbox):
    # frame is type numpy array (imagen RGB) 
    # bbox is a list = [x0,y0, x1,y1]   
    return frame[y0:y1, x0,x1] 


def correct_dict_bboxes(dict_bboxes, width, height):
    if not len(list(dict_bboxes.keys()))>0:
        return None
    for k in list(dict_bboxes.keys()):
        x0,y0,x1,y1 = dict_bboxes[k]
        if x0<0: x0 = 0
        if y0<0: y0 = 0
        if x1>width : x1 = width
        if y1>height: y1 = height
        dict_bboxes[k] = [int(x0),int(y0),int(x1),int(y1)]
    return dict_bboxes

def multi_add_tracklet_queue(args):
    return add_tracklet_queue(*args)


def multi_tracklets_data_add(args):
    tracklets_data = args[0]
    return tracklets_data.add(*args[1:])




def pesquisa_tracking(  input_video         ,    
                        dir_gallery    = None,
                        dir_log        = None,
                        dir_parameters = None,
                        save_separate  = True ,
                        show_video     = True,
                        pause_video    = False):

    global global_id
    global_id                  = -1
    set_id_global(global_id)

    flag                       = False
    flag_finish                = 'DONE'
    every_n_frame              = 1
    apply_model_every_n_frames = 20
    assert apply_model_every_n_frames%every_n_frame==0, "error apply_model_every_n_frames%every_n_frame==0"
    tam_max_tracklet           = 20


    dict_average_time          = dict()
    num_ids_detection          = 0
    current_frame              = -1
    test_output                = dict()
    number_process_threads     = 7
    name_dict_numframe_time    = "numFrames_time.json"
    name_dict_numframe_time    = os.path.join(dir_parameters,"numFrames_time.json")

    path_video = os.path.basename(input_video)
    name_video = os.path.splitext(path_video)[0]
   

    if dir_gallery is None:
        dir_gallery    = check_dir_exist(dir_gallery, 'trash_gallery')   
    if dir_log is None:     
        dir_log        = check_dir_exist(dir_log, 'log')
    if dir_parameters is None:     
        dir_parameters = check_dir_exist(dir_parameters, 'parameters_video')        
    
    # breakpoint()

    #new add
    global_queue = Queue() 
    myevent      = Event()

    print("input_video  : ", input_video )
    print("dir_gallery  : ", dir_gallery )
    print("dir_log      : ", dir_log     )

    # breakpoint()
    
    myobjdetec      = obj_detection()
    mytracker       = Re3Tracker()
    tracklets_data  = tracklets( tam_max_queue = tam_max_tracklet )

    # breakpoint()

    # thread_to_save = Process( target = ( loop_queue_save ), 
    #                      args   = ( global_queue, myevent, dir_gallery, name_video, save_separate, flag_finish )
    #                     )

    thread_to_save = Process( target = ( loop_queue_save ), 
                              args   = ( global_queue, myevent, dir_gallery, name_video, save_separate, flag_finish )
                             )
    thread_to_save.daemon = True
    thread_to_save.start()


    cap                 = setup_window(input_video, name_video, show_video)
    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("number_total_frames: ", number_total_frames )


    start_time = time.time()
    # breakpoint()

    while (cap.isOpened()):
        ret, frame    = cap.read()
        if ret == False:
            print('erro video')
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # if current_frame%every_n_frame==0:
        #     continue
        time_init_frame = time.time()
        
        if current_frame%every_n_frame==0  :
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ INIT FRAME")
            print("current_frame: ",current_frame)

            frameRGB = frame[:,:,::-1].copy()
            frame_plot = frame.copy()


            ids, initBoundingBox = [], {}
            ids, initBoundingBox = myobjdetec.detect(frameRGB)
            initBoundingBox = correct_dict_bboxes(initBoundingBox, width_cap, height_cap)

            num_ids_detection    = len(ids)
            if num_ids_detection > 0:
                global_id    = max_id(initBoundingBox)
                # a, b, c, d = mytracker.multi_track_v3(ids, frameRGB, initBoundingBox)
                test_output = mytracker.multi_track_v4(ids, frameRGB, initBoundingBox)

                if show_video:
                    for key, bbox in test_output.items():
                        # out = tracklets_data.add(current_frame, frame, key, bbox)
                        draw_bbox(frame_plot, key, test_output, bbox)

                ## normal
                # for k,v in initBoundingBox.items():
                #     out = tracklets_data.add(current_frame, frame, k, v)
                ## paralell
                list_keys           = list(initBoundingBox.keys())                
                list_bbox           = list(initBoundingBox.values())
                list_current_frame  = [current_frame  for _ in range(len(list_keys))]
                list_frame          = [frame          for _ in range(len(list_keys))]
                list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                pool = mp.Pool(number_process_threads)
                out  = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_current_frame, list_frame, list_keys, list_bbox) )
                pool.close()
                pool.join()
                
            if  show_window(name_video, frame_plot, show_video, pause_video) is False:
                break

            # else:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break    
            


            # if len(test_output)>0:
            #     if show_video:
            #         for key, bbox in test_output.items():
            #             # out = tracklets_data.add(current_frame, frame, key, bbox)
            #             draw_bbox(frame_plot, key, test_output, bbox)

            #     list_keys           = list(test_output.keys())                
            #     list_bbox           = list(test_output.values())
            #     list_current_frame  = [current_frame  for _ in range(len(list_keys))]
            #     list_frame          = [frame          for _ in range(len(list_keys))]
            #     list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
            #     pool = mp.Pool(number_process_threads)
            #     out  = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_current_frame, list_frame, list_keys, list_bbox) )
            #     pool.close()
            #     pool.join()


            #     if  show_window(name_video, frame_plot, show_video, pause_video) is False:
            #         break

            # else:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break    



            break

            time_end_frame = time.time()
            dict_average_time[current_frame] = time_end_frame-time_init_frame
            print("Time frame : {:.6f}".format( time_end_frame-time_init_frame))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")

    # breakpoint()
  


    while(cap.isOpened()):
        ret, frame    = cap.read()
        if ret == False: 
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_init_frame = time.time()
        
        if current_frame%every_n_frame==0 :
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ INIT FRAME")
            print("current_frame: ",current_frame)

            frameRGB = frame[:,:,::-1].copy()
            frame_plot = frame.copy()

            tracklets_full = tracklets_data.get_list_ids_tracklets_full()
            if len(tracklets_full) > 0:
                ## normal
                # for i in tracklets_data.get_list_ids_tracklets_full():
                #     out = add_tracklet_queue(global_queue, tracklets_data.get_tracklet(i) )
                
                ## paralell             
                list_queue           = [global_queue for _ in range(len(tracklets_full))]
                list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in tracklets_full]
                
                # list_ids             = tracklets_data.get_list_ids_tracklets_full()              
                # list_queue           = [global_queue for _ in range(len(list_ids))]
                # list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in list_ids]

                pool = mp.Pool(number_process_threads)
                out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_get_tracklets) )
                pool.close()
                pool.join()
                flag = True
            
            num_ids_detection = len( tracklets_data.get_list_ids_tracklets() ) 
            if num_ids_detection==0:
                flag = True


            if (flag) or (current_frame%apply_model_every_n_frames==0) :
                flag = False
                ids, initBoundingBox, ids_updates = [], {}, []
                ids, initBoundingBox              = myobjdetec.detect(frameRGB)
                initBoundingBox                   = correct_dict_bboxes(initBoundingBox, width_cap, height_cap)

                new_detections                    = len(ids) 
                if new_detections > 0:
                    # print("    ->  merge_initBoundingBox_v3")
                    dict_ids_bboxs                    = tracklets_data.get_dict_ids_bboxs_tail()
                    ids, initBoundingBox, ids_updates = merge_initBoundingBox_v3(dict_ids_bboxs, initBoundingBox)
                    mytracker.clear_multi_tracking()
                    # bboxes, test_output, _, ids_out = mytracker.multi_track_v3(ids, frameRGB, initBoundingBox)
                    test_output = mytracker.multi_track_v4(ids, frameRGB, initBoundingBox)

                    ids      = set(ids)
                    ids_out  = set(test_output.keys())
                    save_ids = ids.difference(ids_out)
                    
                    # breakpoint()
                    if len(save_ids)>0:
                        ## normal
                        # for i in save_ids:
                        #     out = add_tracklet_queue(global_queue, tracklets_data.get_tracklet(i) )
                        
                        ## paralell
                        list_ids             = list(save_ids)               
                        list_queue           = [global_queue for _ in range(len(list_ids))]
                        list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in list_ids]
                        pool = mp.Pool(number_process_threads)
                        out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_get_tracklets) )
                        pool.close()
                        pool.join()

                    ## normal
                    # for key, bbox in test_output.items():
                    #     out = tracklets_data.add(current_frame, frame, key, bbox)
                    
                    ## paralell
                    list_keys           = list(test_output.keys())                
                    list_bbox           = list(test_output.values())
                    list_current_frame  = [current_frame  for _ in range(len(list_keys))]
                    list_frame          = [frame          for _ in range(len(list_keys))]
                    list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                    pool = mp.Pool(number_process_threads)
                    out  = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_current_frame, list_frame, list_keys, list_bbox) )
                    pool.close()
                    pool.join()
                    continue    
                # elif len(tracklets_data.get_list_ids_tracklets()) > 0:
                #     ## normal
                #     # for i in tracklets_data.get_list_ids_tracklets():
                #     #     out = add_tracklet_queue(global_queue, tracklets_data.get_tracklet(i) )
                    
                #     ## paralell
                #     list_ids             = tracklets_data.get_list_ids_tracklets()                
                #     list_queue           = [global_queue  for _ in range(len(list_ids))]
                #     list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in list_ids]
                #     pool = mp.Pool(number_process_threads)
                #     out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_get_tracklets) )
                #     pool.close()
                #     pool.join()
                
            num_ids_detection = len( tracklets_data.get_list_ids_tracklets() ) 
            if num_ids_detection > 0:
                try:
                    ## REVISAR EN EL CASO QUE ENTRE UN SOLO BOUNDING BOX
                    ids         = tracklets_data.get_list_ids_tracklets()
                    test_output = mytracker.multi_track_v4(ids, frameRGB)
                    # bboxes, test_output, _, ids_out = mytracker.multi_track_v3(ids, frameRGB)
                    
                    ids      = set(ids)
                    ids_out  = set(test_output.keys())
                    save_ids = ids.difference(ids_out)
                    
                    if len(save_ids)>0:
                        ##normal
                        # for i in save_ids:
                        #     out = add_tracklet_queue(global_queue, tracklets_data.get_tracklet(i) )
                       
                        ## paralell
                        list_ids             = list(save_ids)                
                        list_queue           = [global_queue for _ in range(len(list_ids))]
                        list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in list_ids]
                        pool = mp.Pool(number_process_threads)
                        out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_get_tracklets) )
                        pool.close()
                        pool.join()



                except ValueError:
                    # print("             ------->>>>>______ error multi_track_v3")
                    print("             ------->>>>>______ error multi_track_v4")

                if len(test_output)>0:
                    if show_video:
                        for key, bbox in test_output.items():
                            # out = tracklets_data.add(current_frame, frame, key, bbox)
                            draw_bbox(frame_plot, key, test_output, bbox)

                    list_keys           = list(test_output.keys())                
                    list_bbox           = list(test_output.values())
                    list_current_frame  = [current_frame  for _ in range(len(list_keys))]
                    list_frame          = [frame          for _ in range(len(list_keys))]
                    list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                    pool = mp.Pool(number_process_threads)
                    out  = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_current_frame, list_frame, list_keys, list_bbox) )
                    pool.close()
                    pool.join()


                if  show_window(name_video, frame_plot, show_video, pause_video) is False:
                    break

            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break      


            time_end_frame = time.time()
            dict_average_time[current_frame] = time_end_frame-time_init_frame
            print("Time frame : {:.6f}".format( time_end_frame-time_init_frame))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")


    # breakpoint()

    ## normal
    # for i in tracklets_data.get_list_ids_tracklets():
    #     out = add_tracklet_queue(global_queue, tracklets_data.get_tracklet(i) )
    
    ## paralell
    list_ids             = tracklets_data.get_list_ids_tracklets()               
    list_queue           = [global_queue for _ in range(len(list_ids))]
    list_get_tracklets   = [tracklets_data.get_tracklet(i) for i in list_ids]
    pool = mp.Pool(number_process_threads)
    out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_get_tracklets) )
    pool.close()
    pool.join()

    # breakpoint()


    end_time = time.time()
    print("tiempo de demora")
    print(end_time-start_time)

    cap.release()
    cv2.destroyAllWindows()
    global_queue.put(flag_finish)
    thread_to_save.join()

    with open(name_dict_numframe_time, "w") as outfile: 
        json.dump(dict_average_time, outfile, indent = 4) 

    values_time = [float(i) for i in list(dict_average_time.values()) ]
    avg_time    = np.mean(np.asarray(values_time))
    print("average time frame by frame: {:.5f}".format(avg_time))

    return 


def generate_tracklets(input_video, path_results):

    # version_pack = 0 # debug
    version_pack = 1 # final
    i=0

    dir_vesion = 'pack_v{}.{}'
    path_pack = os.path.join(path_results, dir_vesion.format(version_pack, i))
    while(os.path.exists(path_pack)):
        path_pack = os.path.join(path_results, dir_vesion.format(version_pack, i))
        i+=1

    create_dir(path_pack)

    ###################################################

    dir_gallery     = os.path.join(path_pack,'cropping')
    dir_log         = os.path.join(path_pack,'log')
    dir_parameters  = os.path.join(path_pack,'parameters_video')

    create_dir(dir_gallery)
    create_dir(dir_log)
    create_dir(dir_parameters)


    pesquisa_tracking(  input_video,
                        dir_gallery, 
                        dir_log,
                        dir_parameters,
                        save_separate = True,
                        show_video    = True,
                        # pause_video   = False)
                        pause_video   = True)




# Main testing 
if __name__ == "__main__":
    
      
    # video_path   = '/home/luigy/luigy/develop/FF-PRID-2020/RW-PRID-01/A-B/000001/video_in_32-42seg.avi'
    # video_path   = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_10seg.avi'
    video_path   = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_2seg.avi'
    results_path = '/home/luigy/luigy/develop/re3/tracking/tracklet/results'

    generate_tracklets(video_path, results_path)
