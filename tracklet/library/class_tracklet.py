# from queue import Queue
from multiprocessing.dummy import Process, Event, Queue

import numpy as np

msg_error = "error - {}"


def croping_frame(frame, bbox):
    # frame is type numpy array (imagen RGB) 
    # bbox is a list = [x0,y0, x1,y1]
    x0, y0, x1, y1   = bbox
    x0, y0, x1, y1   = int(x0),int(y0), int(x1), int(y1)
    crop             = frame.copy()
    crop             = crop[y0:y1, x0:x1]
    return crop

def crop_to_int(bbox):
    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = int(x0),int(y0), int(x1), int(y1)
    return [x0, y0, x1, y1]

def check_between(value, begin, end):
    return (begin<=value and value<=end)

def area_bbox(bbox):
    width  = abs(int(bbox[2]-bbox[0]))
    height = abs(int(bbox[3]-bbox[1]))
    return  width*height

class track:
    def __init__(self, id_frame, crop_frame, id_bbox, bbox):
        self._id_frame = id_frame
        self._id_bbox  = id_bbox
        self._bbox     = crop_to_int(bbox)
        self._crop     = crop_frame.copy()
    
    def get_id_frame(self):
        # return int(self._id_frame)
        return self._id_frame
    
    def get_crop(self):
        return self._crop
    
    def get_id_bbox(self):
        # return int(self._id_bbox)
        return self._id_bbox
    
    def get_bbox(self):
        return self._bbox

    
class queue_track:
    def __init__(self, id_frame, crop_frame, id_bbox, bbox, tam_max_queue=20):
        self._tam_max_queue = tam_max_queue
        self._tam_track     = 0
        self._queue_tracks  = Queue()
        self._flag_full     = False
        self._init_bbox     = bbox.copy()
        self._width_init    = abs(self._init_bbox[2]-self._init_bbox[0])
        self._height_init   = abs(self._init_bbox[3]-self._init_bbox[1])
        self._add_track(id_frame, crop_frame, id_bbox, bbox)
    
    def set_full(self, flag=True):
        self._flag_full = flag
        return
        
    def __getitem__(self, key):   
        try:
            return self._queue_tracks.queue[key]
        except:
            _msg_error = 'no existe {}'.format(key)
            _msg_error = msg_error.format(_msg_error)
            print(_msg_error)
            return _msg_error
    
    def is_empty(self):
        return self._queue_tracks.empty()
    
    def is_full(self):
        return True if (self._flag_full) or (self._tam_track >= self._tam_max_queue) else False
        
    def _add_track(self, id_frame, crop_frame, id_bbox, bbox ):
        self._queue_tracks.put(track(id_frame, crop_frame, id_bbox, bbox))
        self._tam_track += 1
        return 

    def add_track(self, id_frame, crop_frame, id_bbox, bbox ):
        if not self.is_full():
            # if self.validation_bbox(bbox):
            if area_bbox(bbox)>0:
                self._queue_tracks.put(track(id_frame, crop_frame, id_bbox, bbox))
                self._tam_track += 1
            else:
                self.set_full(True)
            return True
        return False
    
    def get_head_track(self):
        if self._queue_tracks.empty():
            _msg_error = msg_error.format('no hay elementos')
            print(_msg_error)
            return _msg_error
        self._tam_track-=1
        return self._queue_tracks.get()

    def size(self):
        return self._tam_track

    def validation_bbox(self, bbox_new):
        min_w  = 0.6; max_w  = 2.5
        min_h  = 0.5; max_h  = 1.5
        w_crop = abs(bbox_new[2] - bbox_new[0])/self._width_init
        h_crop = abs(bbox_new[3] - bbox_new[1])/self._height_init
        if check_between(w_crop,min_w,max_w) and check_between(h_crop,min_h,max_h):
            return True
        return False

    # def show_queue(self, id):
    #     return 
    
    
    
class tracklets:
    def __init__(self, tam_max_queue=20):
        self._tam_max_queue   = tam_max_queue
        self._dict_tracklets  = dict()
        self._dict_last_frame = dict()
    
    def __getitem__(self, key):
        try:
            return self._dict_tracklets[key]
        except:
            _msg_error = 'no existe {}'.format(key)
            _msg_error = msg_error.format(_msg_error)
            print(_msg_error)
            return _msg_error
    
    def num_tracklets(self):
        return len(self._dict_tracklets)
    
    def get_list_ids_tracklets(self):
        return list(self._dict_tracklets.keys())
    
    def exist_tracklet(self, key):
        return (key in self._dict_tracklets)
    
    def _new_tracklet(self, id_frame, frame, id_bbox, bbox):
        self._dict_last_frame[id_bbox] = frame.copy()
        crop_frame                     = croping_frame(frame, bbox)
        q_track                        = queue_track(id_frame, crop_frame, id_bbox, bbox, tam_max_queue = self._tam_max_queue)
        self._dict_tracklets[id_bbox]  = q_track
        return  True
    
    def _add_tracklet(self, id_frame, frame, id_bbox, bbox):
        self._dict_last_frame[id_bbox] = frame.copy()
        crop_frame                     = croping_frame(frame, bbox)
        self._dict_tracklets[id_bbox].add_track(id_frame, crop_frame, id_bbox, bbox)
        return True
    

    def add(self, id_frame, frame, id_bbox, bbox):
        # if not isinstance(id_frame,int): return False
        if not isinstance(frame, np.ndarray): return False
        # if not isinstance(id_bbox,int): return False
        if not isinstance(bbox, list): return False

        if self.exist_tracklet(id_bbox):
            if self._dict_tracklets[id_bbox].is_full():
                return False
            self._add_tracklet(id_frame, frame, id_bbox, bbox)
            return True
        elif not self.exist_tracklet(id_bbox):
            self._new_tracklet(id_frame, frame, id_bbox, bbox)
            return True
        else:
            print("error tracklet.add")
            return
        
    def remove_tracklet(self, key):
        try:
            last_frame = self._dict_last_frame[key]
            tracklet   = self._dict_tracklets[key]
            del self._dict_last_frame[key]
            del self._dict_tracklets[key]
            return tracklet, last_frame
        except:
            print(msg_error.format("error remove tracklet"))


    def get_lenght_tracklets(self):
        _ids         = self.get_list_ids_tracklets()
        _list_return = list()
        for i in _ids:
            tmp = self._dict_tracklets[i].size()
            _list_return.append(tmp)
        return dict(zip(_ids, _list_return))

    
    def pop_tracklet(self, key):
        try:
            if not self.exist_tracklet(key):
                print(msg_error.format("no se encontro tracklet"))
                return
            tmp_tracklet, tmp_last_frame = self.remove_tracklet(key) 
        except:
            _msg_error = msg_error.format("error pop_tracklet")
            print(_msg_error)
            return _msg_error

        list_return = list()
        while not tmp_tracklet.is_empty():
            tmp      = tmp_tracklet.get_head_track()
            _id_img  = tmp.get_id_frame()
            _img     = tmp.get_crop()
            _id_bbox = tmp.get_id_bbox()
            _bbox    = tmp.get_bbox()
            list_return.append([_id_img, _img, _id_bbox, _bbox])
        return list_return


    def get_tracklet(self, key):
        try:
            if not self.exist_tracklet(key):
                print(msg_error.format("no se encontro tracklet"))
                return
            tmp_tracklet   = self._dict_tracklets[key]
            tmp_last_frame = self._dict_last_frame[key]
        except:
            _msg_error = msg_error.format("error get_tracklet")
            print(_msg_error)
            return _msg_error

        list_return = list()
        while not tmp_tracklet.is_empty():
            tmp      = tmp_tracklet.get_head_track()
            _id_img  = tmp.get_id_frame()
            _img     = tmp.get_crop()
            _id_bbox = tmp.get_id_bbox()
            _bbox    = tmp.get_bbox()
            list_return.append([_id_img, _img, _id_bbox, _bbox])
        return list_return
    


    def get_list_tracklets_full(self):
        assert len(self.get_list_ids_tracklets())>0
        list_return = list()
        for i in self.get_list_ids_tracklets():
            if self._dict_tracklets[i].is_full():
                list_return.append(self.get_tracklet(i))
        return list_return
    

    def get_list_ids_tracklets_full(self):
        list_return = list()
        if not len(self.get_list_ids_tracklets())>0:
            return list_return
        for i in self.get_list_ids_tracklets():
            if self._dict_tracklets[i].is_full():
                list_return.append(i)
        return list_return
        

    def get_dict_ids_bboxs_head(self):
        _list_ids   = self.get_list_ids_tracklets()
        dict_return = dict()
        if len(_list_ids)>0:
            for i in _list_ids:
                dict_return[i] = self._dict_tracklets[i][0].get_bbox()
            return dict_return
        else:
            return False

    def get_dict_ids_bboxs_tail(self):
        _list_ids   = self.get_list_ids_tracklets()
        dict_return = dict()
        if len(_list_ids)>0:
            for i in _list_ids:
                dict_return[i] = self._dict_tracklets[i][-1].get_bbox()
            return dict_return
        else:
            return dict_return


    # def show_track(self,id):
    #     return