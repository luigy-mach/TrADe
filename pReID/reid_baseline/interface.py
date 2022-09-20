import os
import sys
import torch
import torchvision.transforms as T
import shutil
import numpy as np
import cv2
import time

from torch.backends import cudnn
from PIL import Image, ImageDraw
from os.path import join as osjoin
from os.path import split as ossplit


# sys.path.append('..')
# sys.path.append('.')
from .model_reidbaseline         import Backbone
from .utils_reidbaseline.metrics import cosine_similarity, euclidean_distance, cosine_distance
from .utils_reidbaseline.logger  import setup_logger
from .config                     import Config
from .datasets.make_dataloader   import torch_transforms


# torch.cuda.set_per_process_memory_fraction(0.8)


class Reid_baseline():

    # def __init__(self, num_classes=255, top_k=10, mode='val'):
    def __init__(self, num_classes=255, mode='val'):
        self.config      = Config()
        self.num_classes = num_classes
        self.mode        = mode
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.DEVICE_ID
        cudnn.benchmark                    = True

        self.model = Backbone(self.num_classes, self.config)
        self.model.load_param(self.config.TEST_WEIGHT)

        self.device        = 'cuda'
        self.model         = self.model.to(self.device)
        self.transform_val = torch_transforms(self.mode, self.config)
        # self.top_k         = top_k




    def predict(self, query_path, gallery_path, return_time=False, return_best=None):
       
        query_img   = self.open_img(query_path)
        query_input = self.prepare_img(query_img, self.transform_val)
        query_input = query_input.to(self.device)

        self.model.eval()
        with torch.no_grad():
            query_feat = self.model(query_input)

        if return_time:
            list_times  = list()
        list_feats     = list()
        list_imgs_path = list()
        gallery_imgs   = sorted(list(os.listdir(gallery_path)))

        if len(gallery_imgs)==0:
            return [], []

        for file_img in gallery_imgs:
            if not(file_img.endswith(('.png','.jpg','.jpeg'))):
                continue
            # print(file_img)
            time_predic_tinit = time.time()
            abspath_img       = osjoin(gallery_path, file_img)
            img               = self.open_img(abspath_img)
            img_input         = self.prepare_img(img, self.transform_val)
            img_input         = img_input.to(self.device)
            with torch.no_grad():
                feat = self.model(img_input)
            time_predict      = time.time()-time_predic_tinit
            list_feats.append(feat.clone())
            list_imgs_path.append(abspath_img)
            del feat
            torch.cuda.empty_cache()
    
            if return_time:
                list_times.append(time_predict)

        g_feats    = torch.cat(list_feats, dim = 0)

        dist_mat = cosine_similarity(query_feat.clone(), g_feats)
        del query_feat
        torch.cuda.empty_cache()

        # dist_mat = euclidean_distance(query_feat, g_feats)
        # dist_mat = cosine_distance(query_feat, g_feats)
        # breakpoint()
        indices  = np.argsort(dist_mat, axis = 1)
        indices  = indices[0][::-1].reshape((1,-1))
        # if indices.size()>0:
        #     indices  = indices[0][::-1]

        # return np.asarray(list_imgs_path)[indices]
        # breakpoint()


        return1 = np.asarray(list_imgs_path)[indices].reshape((len(list_imgs_path))) 
        return2 = np.take_along_axis(dist_mat, indices, axis=1).reshape((len(list_imgs_path)))

        del list_feats
        del list_imgs_path
        del gallery_imgs

        if return_time:
            return3 = np.asarray(list_times)[indices].reshape((len(list_times))) 
            if return_best:
                if return_best>0:
                    return_best = return_best-1
                return return1[return_best], return2[return_best], np.sum(return3)
            return return1, return2, return3
        
        if return_best:
            if return_best>0:
                return_best = return_best-1
            return return1[return_best], return2[return_best]
        return return1, return2

    def open_img(self, path, rect_width=None, color_line='red'):
        img = Image.open(path)
        if rect_width:
            temp = ImageDraw.Draw(img)
            temp.rectangle((0, 0, img.size[0], img.size[1]), outline=color_line, fill=None, width=rect_width)
        return img

    def prepare_img(self, img, transform):
        query_img = torch.unsqueeze(transform(img), 0)
        return query_img

    def copy_results(self, list_imgs, path_save, top_k, img_size=None, copy_files=True):
        assert top_k>0, 'it needs top_k >0 , currently is {}'.format(top_k) 

        try:
            os.makedirs(path_save, exist_ok=True)
            new_list_imgs = list()
            for i, src_file in enumerate(list_imgs[:top_k]):
                fpath, fname = ossplit(src_file)
                fnewname     = '{:03d}_'.format(i)+fname
                fnewpath     = osjoin(path_save, fnewname)
                new_list_imgs.append(fnewpath)
                if copy_files:
                    shutil.copy2(src_file,fnewpath)
            return np.asarray(new_list_imgs).reshape((-1,1))     
        except:
            print("{} can not be created ", path_save)
            return False

    def _put_text_on_img(self, img, text):
        font          = cv2.FONT_HERSHEY_SIMPLEX
        height, width = img.shape[:2]
        org           = (0, height-10)
        fontScale     = 0.5
        color         = (255, 0, 0) # RGB
        thickness     = 1
        img           = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        return img

    def save_patch_results(self, query_path, list_imgs, dist_mat, path_save, top_k=None, img_size=None, prefix='top', threshold = None):
        top_k_default = 50
        if top_k is None:
            if len(list_imgs)>top_k_default:
                top_k = top_k_default
            else:
                top_k = len(list_imgs)
                
        if img_size==None:
            img_size = self.config.INPUT_SIZE
        self._save_results(query_path, list_imgs, dist_mat, path_save, top_k=top_k, img_size=img_size, prefix=prefix, threshold=threshold)


    def _save_results(self, query_path, list_imgs, dist_mat, path_save, top_k, img_size, prefix, threshold ):
        path, file = ossplit(query_path)
        query_img  = self.open_img(query_path, rect_width = 3)
        figure     = np.asarray(query_img.resize((img_size[1],img_size[0])))
        # breakpoint()
        figure     = self._put_text_on_img(figure,'Query')
        num_imgs   = len(list_imgs)
        top_k      = num_imgs if top_k>num_imgs else top_k

        for k in range(top_k):

            img    = np.asarray(self.open_img(list_imgs[k]).resize((img_size[1],img_size[0])))
            # breakpoint()
            img    = self._put_text_on_img(img,'{:.4f}'.format(dist_mat[k]))
            # img    = np.asarray(self.open_img(list_imgs[0][k]).resize((img_size[1],img_size[0])))
            # if threshold<=dist_mat[k]:
            #     rectangle = np.zeros((img_size[0],10,3), dtype=np.uint8) 
            #     rectangle[:,:,1] = 255 # BGR
            #     figure = np.hstack((figure, rectangle, img))
            if threshold:
                if threshold>=dist_mat[k]:
                    rectangle = np.zeros((img_size[0],10,3), dtype=np.uint8) 
                    rectangle[:,:,1] = 255 # BGR
                    figure = np.hstack((figure, img, rectangle))
                else:
                    figure = np.hstack((figure, img))


        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        fname , ext = os.path.splitext(file)
        if threshold:
            cv2.imwrite(osjoin(path_save, "patch_{}_{}_{}_thr_{}{}".format(fname,prefix,top_k,threshold,ext)),figure)
        else:
            cv2.imwrite(osjoin(path_save, "patch_{}_{}_{}{}".format(fname,prefix,top_k,ext)),figure)

