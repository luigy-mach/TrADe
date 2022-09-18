

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

import multiprocessing.dummy as mp

# basedir = os.path.dirname(__file__)
# sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))

from .object_detector.interface_yolov3_tf import obj_detection as obj_detection

from .object_detector.interface_yolov3_tf import obj_detection as obj_detection_normal
from .object_detector.interface_yolov3_tf_FFPRID import obj_detection as obj_detection_FFPRID

from .multitracker.re3.tracker.re3_tracker import  Re3Tracker
from .library.class_tracklet import *
from .library.parallel_save import *
from .utils_tracklet.bbox import *
from .utils_tracklet.directory import *
from .utils_tracklet.video import *



# ffmpeg -i video_in_45seg.avi -ss 00:00:30 -t 00:00:4 -async 1 -c copy video_in_4seg_error.avi 



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





def correct_dict_bboxes(dict_bboxes, width, height):
    if not len(list(dict_bboxes.keys()))>0:
        return dict()
    for k in list(dict_bboxes.keys()):
        x0,y0,x1,y1      = dict_bboxes[k]
        if x0<0: x0      = 0
        if y0<0: y0      = 0
        if x1>width : x1 = width
        if y1>height: y1 = height
        dict_bboxes[k]   = [int(x0),int(y0),int(x1),int(y1)]
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
                        max_tracklet   = None,
                        save_separate  = True ,
                        show_video     = True,
                        pause_video    = False,
                        debug          = False,
                        every_n_frame  = 1,
                        objDect        = 'yolov3'):

    global global_id
    global_id                  = -1
    set_id_global(global_id)

    flag                       = False
    flag_finish                = 'DONE'
    every_n_frame              = every_n_frame
    if max_tracklet is None:
        apply_model_every_n_frames = 20
        tam_max_tracklet           = 20
    else:
        apply_model_every_n_frames = max_tracklet
        tam_max_tracklet           = max_tracklet

    # assert apply_model_every_n_frames%every_n_frame==0, "error apply_model_every_n_frames%every_n_frame==0"


    dict_average_time       = dict()
    num_ids_detection       = 0
    n_curr_frame            = -1
    # test_output           = dict()
    number_process_threads  = 7
    name_dict_numframe_time = "numFrames_time.json"
    name_dict_numframe_time = os.path.join(dir_parameters,"numFrames_time.json")

    path_video              = os.path.basename(input_video)
    name_video              = os.path.splitext(path_video)[0]
   

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

    if   objDect == 'yolov3':
        myobjdetec = obj_detection_normal()
    elif objDect == 'yolov3-FFPRID':
        myobjdetec = obj_detection_FFPRID()

    mytracker       = Re3Tracker()
    tracklets_data  = tracklets( tam_max_queue = tam_max_tracklet )

    # thread_to_save = Process( target = ( loop_queue_save ), 
    #                      args   = ( global_queue, myevent, dir_gallery, name_video, save_separate, flag_finish )
    #                     )

    thread_to_save = Process( target = ( loop_queue_save ), 
                              # args   = ( global_queue, myevent, dir_gallery, save_separate, flag_finish, name_video )
                              args   = ( global_queue, myevent, dir_gallery, save_separate, flag_finish )
                             )
    thread_to_save.daemon = True
    thread_to_save.start()


    cap                 = setup_window(input_video, name_video, show_video)
    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("number_total_frames: ", number_total_frames )


    general_start_time         = time.time()
    general_process_image_time = 0.0

    # ret, frame    = cap.read()
    # show_window(name_video, frame, show_video, pause_video)

    while (cap.isOpened()):
        time_open_frame_init    = time.time()
        ret, frame              = cap.read()
        show_window(name_video, frame, show_video, pause_video)
        if ret == False:
            print('error video')
            break
        time_open_frame             = time.time() - time_open_frame_init
        general_process_image_time += time_open_frame 

        n_curr_frame    = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_init_frame = time.time()

        if n_curr_frame%every_n_frame==0  :
            if debug:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ INIT FRAME")
                print("current_frame: ",n_curr_frame)

            time_open_image_init         = time.time()
            frameRGB                     = frame[:,:,::-1].copy()
            frame_plot                   = frame.copy()
            time_open_image1             = time.time() - time_open_image_init
            general_process_image_time  += time_open_image1

            _, initBoundingBox = list(), dict()
            _, initBoundingBox = myobjdetec.detect(frameRGB)
            # print("initBoundingBox: ", initBoundingBox)
            initBoundingBox    = correct_dict_bboxes(initBoundingBox, width_cap, height_cap)

            num_ids_detection  = len(initBoundingBox)
            # print("number YOLO detect: ",num_ids_detection)

            if num_ids_detection > 0:
                global_id = max_id(initBoundingBox)
                init_ids  = list(range(1,num_ids_detection+1))
                #test_output é só pra obter os Bboxes e plotar-los
                # breakpoint()
                if show_video:
                    test_output  = mytracker.multi_track_v4(init_ids, frameRGB, initBoundingBox)
                    for key, bbox in test_output.items():
                        # out = tracklets_data.add(n_curr_frame, frame, key, bbox)
                        draw_bbox(frame_plot, key, test_output, bbox)
                else:
                    _  = mytracker.multi_track_v4(init_ids, frameRGB, initBoundingBox)

                ## normal
                # for k,v in initBoundingBox.items():
                #     out = tracklets_data.add(n_curr_frame, frame, k, v)
                ## paralell
                list_keys           = list(initBoundingBox.keys())
                list_bbox           = list(initBoundingBox.values())
                list_n_curr_frame   = [n_curr_frame   for _ in range(len(list_keys))]
                list_frame          = [frame          for _ in range(len(list_keys))]
                list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                pool                = mp.Pool(number_process_threads)
                out                 = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_n_curr_frame, list_frame, list_keys, list_bbox) )
                pool.close()
                pool.join()
                
                if  show_window(name_video, frame_plot, show_video, pause_video) is False:
                    break

                break
            # else:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break    
            


            # if len(test_output)>0:
            #     if show_video:
            #         for key, bbox in test_output.items():
            #             # out = tracklets_data.add(n_curr_frame, frame, key, bbox)
            #             draw_bbox(frame_plot, key, test_output, bbox)

            #     list_keys           = list(test_output.keys())                
            #     list_bbox           = list(test_output.values())
            #     list_n_curr_frame  = [n_curr_frame  for _ in range(len(list_keys))]
            #     list_frame          = [frame          for _ in range(len(list_keys))]
            #     list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
            #     pool = mp.Pool(number_process_threads)
            #     out  = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_n_curr_frame, list_frame, list_keys, list_bbox) )
            #     pool.close()
            #     pool.join()


            #     if  show_window(name_video, frame_plot, show_video, pause_video) is False:
            #         break

            # else:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break    


            # breakpoint()
            

            time_end_frame                  = time.time()
            total_time_frame                = time_end_frame - time_init_frame - time_open_image1
            dict_average_time[n_curr_frame] = total_time_frame
            if debug:
                print("Time frame : {:.6f}".format( total_time_frame))
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")
  

     

    while(cap.isOpened()):
        ret, frame    = cap.read()
        if ret == False: 
            break
        n_curr_frame    = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        time_init_frame = time.time()
        n_curr_frame
        if n_curr_frame%every_n_frame==0 :
            if debug:
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ INIT FRAME")
                print("current_frame: ",n_curr_frame)



            time_open_image_init          = time.time()
            frameRGB                      = frame[:,:,::-1].copy()
            frame_plot                    = frame.copy()
            time_open_image2              = time.time() - time_open_image_init
            general_process_image_time   += time_open_image2

            tracklets_full                = tracklets_data.get_list_ids_tracklets_full()
            if len(tracklets_full) > 0:
                ## normal
                # for i in tracklets_data.get_list_ids_tracklets_full():
                #     out = add_tracklet_queue(global_queue, tracklets_data.pop_tracklet(i) )
                
                ## paralell             
                list_queue           = [global_queue                   for _ in range(len(tracklets_full))]
                list_pop_tracklets   = [tracklets_data.pop_tracklet(i) for i in tracklets_full]
                
                # list_ids           = tracklets_data.get_list_ids_tracklets_full()
                # list_queue         = [global_queue for _ in range(len(list_ids))]
                # list_pop_tracklets = [tracklets_data.pop_tracklet(i) for i in list_ids]

                pool                 = mp.Pool(number_process_threads)
                out                  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_pop_tracklets) )
                pool.close()
                pool.join()
                # flag = True
            
            num_ids_detection = len( tracklets_data.get_list_ids_tracklets() ) 
            if num_ids_detection==0:
                flag = True

            if (flag) or (n_curr_frame%apply_model_every_n_frames==0) :
                # print(">>>>>>>>>>>>>>>>>>>>>>>> NEW DETECTION INIT")
                flag = False
                _, new_initBoundingBox, ids_updates = list(), dict(), list()
                _, new_initBoundingBox              = myobjdetec.detect(frameRGB)
                new_initBoundingBox                 = correct_dict_bboxes(new_initBoundingBox, width_cap, height_cap)

                num_new_detections                  = len(new_initBoundingBox) 
                # print("number YOLO detect: ", num_new_detections)
                # print(">>>>>>>>>>>>>>>>>>>>>>>> NEW DETECTION object")
                if num_new_detections > 0:
                    # print("    ->  merge_initBoundingBox_v3")

                    dict_ids_bboxs                               = tracklets_data.get_dict_ids_bboxs_tail()
                    ids_old                                      = list(dict_ids_bboxs.keys())
                    ids_new, update_initBoundingBox, ids_updates = merge_initBoundingBox_v3(dict_ids_bboxs, new_initBoundingBox)
                    mytracker.clear_multi_tracking()
                    
                    # if len(ids_new)!=len(update_initBoundingBox.values()):
                    #     breakpoint()
                    test_output = mytracker.multi_track_v4(ids_new, frameRGB, update_initBoundingBox)

                    # print(">>>>>>>>>>>>>>>>>>>>>>>> NEW DETECTION END")
                    ids_old     = set(ids_old)
                    ids_out     = set(test_output.keys())
                    save_ids    = ids_old.difference(ids_out)
                    
                    if len(save_ids)>0:
                        ## normal
                        # for i in save_ids:
                        #     out = add_tracklet_queue(global_queue, tracklets_data.pop_tracklet(i) )
                        
                        ## paralell
                        list_ids           = list(save_ids)
                        list_queue         = [global_queue                   for _ in range(len(list_ids))]
                        list_pop_tracklets = [tracklets_data.pop_tracklet(i) for i in list_ids]
                        pool               = mp.Pool(number_process_threads)
                        out                = pool.map(multi_add_tracklet_queue, zip(list_queue, list_pop_tracklets) )
                        pool.close()
                        pool.join()

                    ## normal
                    # for key, bbox in test_output.items():
                    #     out = tracklets_data.add(n_curr_frame, frame, key, bbox)
                    
                    ## paralell
                    list_keys           = list(test_output.keys())
                    list_bbox           = list(test_output.values())
                    list_n_curr_frame   = [n_curr_frame   for _ in range(len(list_keys))]
                    list_frame          = [frame          for _ in range(len(list_keys))]
                    list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                    pool                = mp.Pool(number_process_threads)
                    out                 = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_n_curr_frame, list_frame, list_keys, list_bbox) )
                    pool.close()
                    pool.join()

                    time_end_frame                   = time.time()
                    time_total_frame                 = time_end_frame - time_init_frame - time_open_image2
                    dict_average_time[n_curr_frame]  = time_total_frame 
                    if debug:
                        print("Time frame : {:.6f}".format(time_total_frame))
                        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")


                    continue    
                # elif len(tracklets_data.get_list_ids_tracklets()) > 0:
                #     ## normal
                #     # for i in tracklets_data.get_list_ids_tracklets():
                #     #     out = add_tracklet_queue(global_queue, tracklets_data.pop_tracklet(i) )
                    
                #     ## paralell
                #     list_ids             = tracklets_data.get_list_ids_tracklets()                
                #     list_queue           = [global_queue  for _ in range(len(list_ids))]
                #     list_pop_tracklets   = [tracklets_data.pop_tracklet(i) for i in list_ids]
                #     pool = mp.Pool(number_process_threads)
                #     out  = pool.map(multi_add_tracklet_queue, zip(list_queue, list_pop_tracklets) )
                #     pool.close()
                #     pool.join()

            # num_ids_detection = len( tracklets_data.get_list_ids_tracklets() ) 
            if num_ids_detection > 0:
                try:
                    ## REVISAR EN EL CASO QUE ENTRE UN SOLO BOUNDING BOX
                    ids         = tracklets_data.get_list_ids_tracklets()
                    ids_tam     = tracklets_data.get_lenght_tracklets()
                    print("TRACKLET and Size : ", ids_tam)
                    
                    # if ids[0]=='15':
                    #     breakpoint()
                    # print("ids tracklets_data   : ", ids) 
                    test_output = mytracker.multi_track_v4(ids, frameRGB)
                    # print("test_output TRACKLET : ", list(test_output.keys()))
                    # bboxes, test_output, _, ids_out = mytracker.multi_track_v3(ids, frameRGB)
                    
                    ids      = set(ids)
                    ids_out  = set(test_output.keys())
                    save_ids = ids.difference(ids_out)
                    
                    if len(save_ids)>0:
                        ##normal
                        # for i in save_ids:
                        #     out = add_tracklet_queue(global_queue, tracklets_data.pop_tracklet(i) )
                       
                        ## paralell
                        list_ids           = list(save_ids)
                        list_queue         = [global_queue                   for _ in range(len(list_ids))]
                        list_pop_tracklets = [tracklets_data.pop_tracklet(i) for i in list_ids]
                        pool               = mp.Pool(number_process_threads)
                        out                = pool.map(multi_add_tracklet_queue, zip(list_queue, list_pop_tracklets) )
                        pool.close()
                        pool.join()



                except ValueError:
                    # print("             ------->>>>>______ error multi_track_v3")
                    print("             ------->>>>>______ error multi_track_v4")

                if len(test_output)>0:
                    if show_video:
                        for key, bbox in test_output.items():
                            # out = tracklets_data.add(n_curr_frame, frame, key, bbox)
                            draw_bbox(frame_plot, key, test_output, bbox)

                    list_keys           = list(test_output.keys())
                    list_bbox           = list(test_output.values())
                    list_n_curr_frame   = [n_curr_frame   for _ in range(len(list_keys))]
                    list_frame          = [frame          for _ in range(len(list_keys))]
                    list_tracklets_data = [tracklets_data for _ in range(len(list_keys))]
                    pool                = mp.Pool(number_process_threads)
                    out                 = pool.map(multi_tracklets_data_add, zip(list_tracklets_data, list_n_curr_frame, list_frame, list_keys, list_bbox) )
                    pool.close()
                    pool.join()


                if  show_window(name_video, frame_plot, show_video, pause_video) is False:
                    break

            
            if show_video:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break      

            if num_ids_detection==0:
                show_window(name_video, frame_plot, show_video, pause_video)


            time_end_frame                  = time.time()
            time_total_frame                = time_end_frame - time_init_frame - time_open_image2
            dict_average_time[n_curr_frame] = time_total_frame
            if debug:
                print("Time frame : {:.6f}".format( time_total_frame))
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")

    # breakpoint()

    ## normal
    # for i in tracklets_data.get_list_ids_tracklets():
    #     out = add_tracklet_queue(global_queue, tracklets_data.pop_tracklet(i) )
    
    ## paralell
    list_ids           = tracklets_data.get_list_ids_tracklets()
    list_queue         = [global_queue for _ in range(len(list_ids))]
    list_pop_tracklets = [tracklets_data.pop_tracklet(i) for i in list_ids]
    pool               = mp.Pool(number_process_threads)
    out                = pool.map(multi_add_tracklet_queue, zip(list_queue, list_pop_tracklets) )
    pool.close()
    pool.join()

    # breakpoint()


    general_end_time = time.time()
    total_time       = general_end_time - general_start_time - general_process_image_time
    print("tiempo de demora")
    print(total_time)
    print("*********** END")
    cap.release()
    cv2.destroyAllWindows()
    global_queue.put(flag_finish)
    thread_to_save.join()

    with open(name_dict_numframe_time, "w") as outfile: 
        json.dump(dict_average_time, outfile, indent = 4) 

    values_time = [float(i) for i in list(dict_average_time.values()) ]
    avg_time    = np.mean(np.asarray(values_time))
    print("average time frame by frame: {:.5f}".format(avg_time))

    del global_queue
    del myevent
    del mytracker
    del myobjdetec
    del tracklets_data
    return total_time



def generate_tracklets(input_video, path_results, max_tracklet=None, packName=None , return_time=False ,show_window=True, objDect='yolov3', every_n_frame=1):

    assert os.path.exists(input_video)
    assert os.path.exists(path_results)

    path_video = os.path.basename(input_video)
    name_video = os.path.splitext(path_video)[0]

    if packName is None:
        path_pack = path_results        
    else:
        path_pack = os.path.join(path_results, packName)
    create_dir(path_pack)

    ###################################################

    dir_gallery     = os.path.join(path_pack,'cropping')
    dir_log         = os.path.join(path_pack,'log')
    dir_parameters  = os.path.join(path_pack,'parameters_video')

    create_dir(dir_gallery)
    create_dir(dir_log)
    create_dir(dir_parameters)


    time = pesquisa_tracking(  input_video,
                                dir_gallery, 
                                dir_log,
                                dir_parameters,
                                max_tracklet  = max_tracklet,
                                save_separate = True,
                                show_video    = show_window,
                                # pause_video   = False)
                                pause_video   = True,
                                debug         = True,
                                every_n_frame = every_n_frame,
                                objDect       = objDect)

    if return_time:
        return time
    else:
        return

def check_bbox(bboxes, video_width, video_height):
    # breakpoint()
    x0, y0, x1, y1 = bboxes
    if x0<0:
        x0=0
    if y0<0:
        y0=0
    if x1>video_width:
        x1=video_width
    if y1>video_height:
        y1=video_height
    return x0, y0, x1, y1


def gallery_yolo(  input_video, dir_gallery,  dir_log, dir_parameters, save_separate=True, show_video=True, objDect='yolov3', every_n_frame=1):

    dict_average_time          = dict()
    name_dict_numframe_time    = "numFrames_time.json"
    name_dict_numframe_time    = os.path.join(dir_parameters, "numFrames_time.json")

    path_video = os.path.basename(input_video)
    name_video = os.path.splitext(path_video)[0]
   

    if dir_gallery is None:
        dir_gallery    = check_dir_exist(dir_gallery, 'trash_gallery')   
    if dir_log is None:     
        dir_log        = check_dir_exist(dir_log, 'log')
    if dir_parameters is None:     
        dir_parameters = check_dir_exist(dir_parameters, 'parameters_video')        
    
    if objDect == 'yolov3':
        myobjdetec          = obj_detection_normal()
    elif objDect == 'yolov3-FFPRID':
        myobjdetec          = obj_detection_FFPRID()

    cap                 = setup_window(input_video, name_video, show_video)
    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("number_total_frames: ", number_total_frames )

    start_time = time.time()


    # ret, frame    = cap.read()
    # show_window(name_video, frame, show_video, pause_video)



    while (cap.isOpened()):
        ret, frame    = cap.read()
        if ret == False:
            # print('erro video')
            break
        n_curr_frame  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if n_curr_frame%every_n_frame==0:
            
            
            frameRGB   = frame[:,:,::-1].copy()
            frame_plot = frame.copy()

            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ INIT FRAME")
            time_init_frame = time.time()

            ids, initBoundingBox = [], {}
            ids, initBoundingBox = myobjdetec.detect(frameRGB)

            if len(initBoundingBox)>0:
                # breakpoint()
                for _id, _bbox in initBoundingBox.items():
                    x0, y0, x1, y1 = check_bbox(_bbox, width_cap, height_cap)

                    _name = "frame_{:06d}_id_{:06d}_bbox_{}_{}_{}_{}.png"
                    _name = _name.format(int(n_curr_frame), int(_id), x0,y0,x1,y1)

                    # if separate:
                    #     _path = os.path.join(path_main,str(_id_bbox))
                    #     create_dir(_path)
                    _crop = frame[y0:y1, x0:x1]
                    _path = os.path.join(dir_gallery, _name)
                    cv2.imwrite(_path, _crop)

            time_end_frame                  = time.time()
            dict_average_time[n_curr_frame] = time_end_frame-time_init_frame

            print("curr frame : {}".format( n_curr_frame))
            print("Time frame : {:.6f}".format( time_end_frame-time_init_frame))
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++ END FRAME")



    end_time = time.time()
    print("*********** END")
    print("total time ")
    print(end_time-start_time)
    print("*********** END")
    cap.release()
    cv2.destroyAllWindows()
   
    with open(name_dict_numframe_time, "w") as outfile: 
        json.dump(dict_average_time, outfile, indent = 4) 

    values_time = [float(i) for i in list(dict_average_time.values()) ]
    values_time = np.asarray(values_time)
    avg_time    = np.mean(values_time)
    print("average time frame by frame: {:.5f}".format(avg_time))

    return np.sum(values_time)


def generate_gallery_yolo(input_video, path_results, show_window=True, max_tracklet=None, packName=False, objDect='yolov3', every_n_frame=1, return_time=False ):

    assert os.path.exists(input_video)
    assert os.path.exists(path_results)

    path_video = os.path.basename(input_video)
    name_video = os.path.splitext(path_video)[0]

    if packName is None: 
        path_pack = path_results        
    else:
        path_pack = os.path.join(path_results, packName)
    create_dir(path_pack)


    ###################################################

    dir_gallery     = os.path.join(path_pack,'gallery_yolo')
    dir_log         = os.path.join(path_pack,'log_yolo')
    dir_parameters  = os.path.join(path_pack,'parameters_video_yolo')

    create_dir(dir_gallery)
    create_dir(dir_log)
    create_dir(dir_parameters)

    # breakpoint() 

    time = gallery_yolo(    input_video,
                            dir_gallery, 
                            dir_log,
                            dir_parameters,
                            save_separate = True,
                            show_video    = show_window,
                            every_n_frame = every_n_frame,
                            objDect       = objDect)
    if return_time:
        return time
    else:
        return




# Main testing 
if __name__ == "__main__":
    
      
    # video_path   = '/home/luigy/luigy/develop/FF-PRID-2020/RW-PRID-01/A-B/000001/video_in_45seg.avi'
    # video_path   = '/home/luigy/luigy/develop/FF-PRID-2020/RW-PRID-01/A-B/000001/video_in_4seg.avi'
    # video_path   = '/home/luigy/luigy/develop/FF-PRID-2020/RW-PRID-01/A-B/000001/video_in_4seg_error.avi'
    # video_path   = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_10seg.avi'
    # video_path   = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_2seg.avi'
    results_path = '/home/luigy/luigy/develop/re3/tracking/tracklet/results'

    # generate_tracklets(video_path, results_path)

    path_main = '/home/luigy/luigy/develop/FF-PRID-2020/RW-PRID-01/all/videos_all'
    for i in os.listdir(path_main):
        video_path = os.path.join(path_main, i)
        print(video_path)
        # generate_tracklets(video_path, results_path)






