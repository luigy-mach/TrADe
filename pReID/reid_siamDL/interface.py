
import os
import shutil
import numpy as np
from os.path import join as osjoin
from os.path import split as ossplit
from PIL import Image, ImageDraw


from .config import Config

###


import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd
# from pYOLO import cuhk03_dataset
# this script has functions for to buiding SiamIDL Network

import sys
basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir, os.path.pardir)))

from utils.handler_file import *

FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_integer('batch_size', '150', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
# tf.flags.DEFINE_string('logs_dir', './logs/', 'path to logs directory')
tf.flags.DEFINE_string('logs_dir', '/home/luigy/luigy/develop/re3/tracking/pReID/logs', 'path to logs directory')
#tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
#tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test, data')
#tf.flags.DEFINE_string('image1', '', 'First image path to compare')
#tf.flags.DEFINE_string('image2', '', 'Second image path to compare')
#tf.flags.DEFINE_string('path_test', '', 'Images path to compare')

tf.flags.DEFINE_string('query_path'    , ''                     , 'First image path to compare')
tf.flags.DEFINE_string('cropps_path'   , ''                     , 'gallery')
tf.flags.DEFINE_integer('top'          , '20'                   , 'Top for reid')



# IMAGE_WIDTH  = 60
# IMAGE_HEIGHT = 160


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


def sortSecond(val):
    return val[1]

def sortFirst(val):
    return val[0]


tf.random.set_random_seed(1234)


class Reid_SiamlDL(object):
    def __init__(self):
        self.config        = Config()
           
        #batch size = 1, solo para TESTS
        self.batch_size    = 1
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.images        = tf.placeholder(tf.float32, [2, self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3], name='images')
        self.labels        = tf.placeholder(tf.float32, [self.batch_size, 2], name='labels')
        self.is_train      = tf.placeholder(tf.bool, name='is_train')
        self.global_step   = tf.Variable(0, name='global_step', trainable=False)
        self.weight_decay  = 0.0005
        #tarin_num_id = 0
        #val_num_id = 0
        self.images1, self.images2 = self.preprocess(self.images, self.is_train)
        print('=======================Build Network=======================')
        self.logits    = self.network(self.images1, self.images2, self.weight_decay)
        self.loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.inference = tf.nn.softmax(self.logits)
        
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
        self.train     = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.lr        = FLAGS.learning_rate

        # self.sess  = tf.Session()
        self.sess = tf.Session(config = tf.ConfigProto( gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40), 
                                                        allow_soft_placement=True)
                               )
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.ckpt  = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            print('==================================Restore model==================================')
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        
        print("FLAGS.logs_dir: ",FLAGS.logs_dir)
        print("FLAGS.learning_rate: ",FLAGS.learning_rate)
        print("FLAGS.max_steps: ", FLAGS.max_steps)


    def predict(self, query_path, gallery_path, return_time=False, return_best=None):

        # list_files   = sorted(os.listdir(gallery_path))
        list_files   = sorted(list(find_files(gallery_path, '*.png', type='absolute')))

        # absPathFiles = [os.path.join(gallery_path,i) for i in list_files]
        absPathFiles = [os.path.join(i) for i in list_files]
        assert os.path.exists(gallery_path), 'err, {} doesnt exist'.format(gallery_path)
        assert len(absPathFiles)>0, 'err, absPathFiles is empty'

        img_query = cv2.imread(query_path)
        img_query = cv2.resize(img_query, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        img_query = np.reshape(img_query, (1, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3)).astype(float)
        

        list_all       = []
        for absPath in absPathFiles:
            time_predic_tinit = time.time()
            img_x      = cv2.imread(absPath)
            img_x      = cv2.resize(img_x, (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))
            img_x      = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
            img_x      = np.reshape(img_x, (1, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3)).astype(float)
            raw_images = np.array([img_query, img_x])
            feed_dict  = {self.images: raw_images, self.is_train: False}
            prediction = self.sess.run(self.inference, feed_dict = feed_dict)
            time_predict      = time.time()-time_predic_tinit
            if return_time:
                tupl       = (absPath, prediction[0][0], prediction[0][1],time_predict)            
            else:
                tupl       = (absPath, prediction[0][0], prediction[0][1])            

            list_all.append(tupl)   
            # if bool(not np.argmax(prediction[0])):
            #     tupl = (absPath, prediction[0][0], prediction[0][1])            
            #     list_all.append(tupl)   
            # else:
            #     print("no entre")
        list_all.sort(key = sortSecond , reverse = True)
        
        #CONTENT list_all was sorted:
        #    ('path','score_reid','score_penalty')

        numpy_all      = np.asarray(list_all)
        # breakpoint()

        paths         = numpy_all[:,0]               # take only paths 
        score_reid    = numpy_all[:,1].astype(float) # take only score_reid
        score_penalty = numpy_all[:,2].astype(float)
        if return_time:
            times = numpy_all[:,3].astype(float)

        # return paths, score_reid


        if return_time:
            if return_best:
                if return_best>0:
                    return_best = return_best-1
                return paths[return_best], score_reid[return_best], np.sum(times)
            return paths, score_reid, times
        
        if return_best:
            if return_best>0:
                return_best = return_best-1
            return paths[return_best], score_reid[return_best]

        return paths, score_reid






    ### add
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
            img    = self._put_text_on_img(img,'{:.4f}'.format(float(dist_mat[k])))
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


    def _put_text_on_img(self, img, text):
        font          = cv2.FONT_HERSHEY_SIMPLEX
        height, width = img.shape[:2]
        org           = (0, height-10)
        fontScale     = 0.5
        color         = (255, 0, 0) # RGB
        thickness     = 1
        img           = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
        return img


    def open_img(self, path, rect_width=None, color_line='red'):
        img = Image.open(path)
        if rect_width:
            temp = ImageDraw.Draw(img)
            temp.rectangle((0, 0, img.size[0], img.size[1]), outline=color_line, fill=None, width=rect_width)
        return img



    def preprocess(self, images, is_train):
        def train():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])
                split[i] = tf.image.resize_images(split[i], [self.config.IMAGE_HEIGHT + 8, self.config.IMAGE_WIDTH + 3])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [self.config.IMAGE_HEIGHT + 8, self.config.IMAGE_WIDTH + 3, 3])
                    split[i][j] = tf.random_crop(split[i][j], [self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])
                    split[i][j] = tf.image.random_flip_left_right(split[i][j])
                    split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                    split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                    split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3]),
                tf.reshape(tf.concat(split[1], axis=0), [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])]
        def val():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])
                split[i] = tf.image.resize_images(split[i], [self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3]),
                tf.reshape(tf.concat(split[1], axis=0), [self.batch_size, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])]
        return tf.cond(is_train, train, val)



    def network(self, images1, images2, weight_decay):
        with tf.variable_scope('network'):
            # Tied Convolution
            conv1_1 = tf.layers.conv2d(images1, 20, [5, 5], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_1')
            pool1_1 = tf.layers.max_pooling2d(conv1_1, [2, 2], [2, 2], name='pool1_1')
            conv1_2 = tf.layers.conv2d(pool1_1, 25, [5, 5], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv1_2')
            pool1_2 = tf.layers.max_pooling2d(conv1_2, [2, 2], [2, 2], name='pool1_2')
            conv2_1 = tf.layers.conv2d(images2, 20, [5, 5], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_1')
            pool2_1 = tf.layers.max_pooling2d(conv2_1, [2, 2], [2, 2], name='pool2_1')
            conv2_2 = tf.layers.conv2d(pool2_1, 25, [5, 5], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='conv2_2')
            pool2_2 = tf.layers.max_pooling2d(conv2_2, [2, 2], [2, 2], name='pool2_2')

            # Cross-Input Neighborhood Differences
            trans   = tf.transpose(pool1_2, [0, 3, 1, 2])
            shape   = trans.get_shape().as_list()
            m1s     = tf.ones([shape[0], shape[1], shape[2], shape[3], 5, 5])
            reshape = tf.reshape(trans, [shape[0], shape[1], shape[2], shape[3], 1, 1])
            f       = tf.multiply(reshape, m1s)

            trans   = tf.transpose(pool2_2, [0, 3, 1, 2])
            reshape = tf.reshape(trans, [1, shape[0], shape[1], shape[2], shape[3]])
            g       = []
            pad     = tf.pad(reshape, [[0, 0], [0, 0], [0, 0], [2, 2], [2, 2]])
            for i in range(shape[2]):
                for j in range(shape[3]):
                    g.append(pad[:,:,:,i:i+5,j:j+5])

            concat   = tf.concat(g, axis                                                               = 0)
            reshape  = tf.reshape(concat, [shape[2], shape[3], shape[0], shape[1], 5, 5])
            g        = tf.transpose(reshape, [2, 3, 0, 1, 4, 5])
            reshape1 = tf.reshape(tf.subtract(f, g), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
            reshape2 = tf.reshape(tf.subtract(g, f), [shape[0], shape[1], shape[2] * 5, shape[3] * 5])
            k1       = tf.nn.relu(tf.transpose(reshape1, [0, 2, 3, 1]), name                           = 'k1')
            k2       = tf.nn.relu(tf.transpose(reshape2, [0, 2, 3, 1]), name                           = 'k2')

            # Patch Summary Features
            l1 = tf.layers.conv2d(k1, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l1')
            l2 = tf.layers.conv2d(k2, 25, [5, 5], (5, 5), activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='l2')

            # Across-Patch Features
            m1 = tf.layers.conv2d(l1, 25, [3, 3], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m1')
            pool_m1 = tf.layers.max_pooling2d(m1, [2, 2], [2, 2], padding='same', name='pool_m1')
            m2 = tf.layers.conv2d(l2, 25, [3, 3], activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), name='m2')
            pool_m2 = tf.layers.max_pooling2d(m2, [2, 2], [2, 2], padding='same', name='pool_m2')

            # Higher-Order Relationships
            concat  = tf.concat([pool_m1, pool_m2], axis             = 3)
            reshape = tf.reshape(concat, [self.batch_size, -1])
            fc1     = tf.layers.dense(reshape, 500, tf.nn.relu, name = 'fc1')
            fc2     = tf.layers.dense(fc1, 2, name                   = 'fc2')

            return fc2



# def main(argv=None):
#     fpath, fname = os.path.split(FLAGS.cropps_path)
#     fpath_reid   = fpath + '/out_reid'
#     if not os.path.isfile(fpath_reid):
#         os.system('mkdir '+ fpath_reid)

#     reidentifier = personReIdentifier()
#     topN         = FLAGS.top
#     print('*******CLASSIC RE-ID TEST*******')
#     reidentifier.PersonReIdentification(FLAGS.query_path, FLAGS.cropps_path, fpath_reid, topN, show_query = True)


# if __name__ == '__main__':
    
#     tf.compat.v1.app.run()






if __name__ == "__main__":

    query_path   = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/query/query_id_13.png'
    # gallery_path = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/pack_v0_top_10_numInlier_1/gallery_inliers'
    gallery_path = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/gallery_inliers_top_1'

    save_path    = '/home/luigy/luigy/develop/re3/tracking/pReID/reid_siamDL/results'

    top_k        = 10
    reidentifier = Reid_SiamlDL()
    reidentifier.predict(query_path, gallery_path, save_path, top_k, show_query = False)

