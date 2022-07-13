import os
from os.path import join as osjoin
from os.path import split as ossplit
import sys
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
# sys.path.append('..')

from utils.logger import setup_logger
from model import make_model
from config import Config

import numpy as np
import cv2
from utils.metrics import cosine_similarity

from datasets.make_dataloader import torch_transforms 



def visualizer(name_file_query, query_img, indices, imgs_path, Cfg, camid='non-define', top_k = 10, img_size=[128,128], path_save=None):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name   = str(indices[0][k]).zfill(6)
        img    = np.asarray(Image.open(imgs_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title  = name

    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    # if not os.path.exists(Cfg.LOG_DIR+ "/results/"):
    #     # print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    #     raise Exception('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))

    if path_save:
        cv2.imwrite(osjoin(path_save, "{}-cam{}.png".format(name_file_query,camid)),figure)
    else:
        cv2.imwrite(Cfg.LOG_DIR+ "/results/{}-cam{}.png".format(name_file_query,camid),figure)


def prepare_img(path_img, transform_val):

    img       = Image.open(path_img)
    query_img = torch.unsqueeze(transform_val(img), 0)
    return img, query_img



if __name__ == "__main__":

    Cfg                                = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = Cfg.DEVICE_ID
    cudnn.benchmark                    = True

    model = make_model(Cfg, 255)
    model.load_param(Cfg.TEST_WEIGHT)

    device        = 'cuda'
    model         = model.to(device)
    transform_val = torch_transforms('val', Cfg)
    log_dir = Cfg.LOG_DIR
    model.eval()
    
    # path_query       = '/home/luigy/luigy/datasets/REID/Market-1501-v15.09.15/query/0001_c5s1_001426_00.jpg'
    # path_query       = '/home/luigy/luigy/datasets/REID/Market-1501-v15.09.15/query/0137_c6s1_024401_00.jpg'
    path_query       = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/query/query_id_13.png'


    query_img, query_input = prepare_img(path_query, transform_val)
    query_input      = query_input.to(device)

    with torch.no_grad():
        query_feat = model(query_input)


    # path_gallery = '/home/luigy/luigy/datasets/REID/Market-1501-v15.09.15/bounding_box_train'
    path_gallery = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/pack_v0_top_10_numInlier_1/gallery_inliers'
    gallery_imgs = sorted(list(os.listdir(path_gallery)))

    list_feats     = list()
    list_imgs_path = list()

    for file_img in gallery_imgs:
        if not(file_img.endswith(('.png','.jpg','.jpeg'))):
            continue
        abspath_img    = osjoin(path_gallery,file_img)
        img, img_input = prepare_img(abspath_img, transform_val)
        img_input      = img_input.to(device)
        with torch.no_grad():
            feat = model(img_input)
            list_feats.append(feat)
            list_imgs_path.append(abspath_img)

    feats    = torch.cat(list_feats, dim = 0)
    dist_mat = cosine_similarity(query_feat, feats)
    indices  = np.argsort(dist_mat, axis = 1)

    path_save = '/home/luigy/luigy/develop/person-reid-tiny-baseline/out_test'
    visualizer(ossplit(path_query)[1], query_img, indices, list_imgs_path, Cfg, path_save=path_save, camid='mixed', top_k=20, img_size=Cfg.INPUT_SIZE)
