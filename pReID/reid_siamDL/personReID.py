


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






IMAGE_WIDTH  = 60
IMAGE_HEIGHT = 160


def sortSecond(val):
    return val[1]

def sortFirst(val):
    return val[0]


# tf.random.set_random_seed(1234)

class personReIdentifier(object):
    def __init__(self):
        #batch size = 1, solo para TESTS
        self.batch_size    = 1
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.images        = tf.placeholder(tf.float32, [2, self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
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

        self.sess  = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.ckpt  = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            print('==================================Restore model==================================')
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        
        print("FLAGS.logs_dir: ",FLAGS.logs_dir)
        print("FLAGS.learning_rate: ",FLAGS.learning_rate)
        print("FLAGS.max_steps: ", FLAGS.max_steps)


    #Improved Deep learning architectura person-ReID
    def PersonReIdentification(self, query_path, cropps_path, out_reid_path, topN, show_query = False):
        print('RE-IDENTIFICATION...............')
        topN     = topN - 1
        #GALLERY , donde se consultara!
        # files    = sorted(glob.glob( cropps_path+'/*.png'))

        files    = sorted(os.listdir(cropps_path))
        files    = [os.path.join(cropps_path,i) for i in files]
        assert os.path.exists(cropps_path), 'err, {} doesnt exist'.format(cropps_path)
        assert len(files)>0, 'err, files is empty'

        print('cropps files: ',len(files))

        image1   = cv2.imread(query_path)
        image1   = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image1   = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        
        if(show_query):
            plt.imshow(image1)
            plt.show()
        
        start    = time.time()
        image1   = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
        
        list_all = []
        for x in files:
            image2      = cv2.imread(x)
            image2      = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2      = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2      = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2])
            feed_dict   = {self.images: test_images, self.is_train: False}
            prediction  = self.sess.run(self.inference, feed_dict = feed_dict)
            
            if bool(not np.argmax(prediction[0])):
                tupl = (x, prediction[0][0], prediction[0][1])            
                list_all.append(tupl)    
        list_all.sort(key = sortSecond , reverse = True)
        

        end = time.time()
        print("Time in seconds: ")
        print(end - start)
        print ("size list predict: ", len(list_all))
        i   = 0# para ordenar, de acuerdo al acierto

        list_reid_coords = []
        list_score = []

        for e in list_all:
            temp_img     = cv2.imread(e[0])
            # estoy leyendo el crop de disco ... para luego escribir en otro lado
            temp_img     = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
            fpath, fname = os.path.split(e[0])
            if (i > topN ):
                break
            #estoy escribiendo ..
            temp_img       = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
            #CARPETA DONDE SE ESCRIBIRA LA SALIDA, LOS RESULTADOS
            cv2.imwrite(out_reid_path+'/'+str(i+1)+'_'+fname, temp_img)
            path_f, name_f = os.path.split(e[0])
            splits_coords  = name_f.rsplit('_')
            last_coord     = splits_coords[5].rsplit('.')
            i              = i +1
            list_reid_coords.append(( int(splits_coords[1]), splits_coords[2], splits_coords[3], splits_coords[4], last_coord[0]))
            #pathi, nameimage = e[0]
            list_score.append((name_f, e[1], e[2]))
            print (i, e[0]," - ", e[1], " - ", e[2])
            
        ## escribo un csv
        df = pd.DataFrame(np.array(list_reid_coords))
        df.to_csv(out_reid_path+"/coords_results.csv", header = False)
        df = pd.DataFrame(np.array(list_score))
        df.to_csv(out_reid_path+"/score_results.csv", header = False)
    

        out_dict = dict()
        for i, ( path, v, s) in enumerate(list_all[:topN+1]):
            mini_dict = split_fpath(path)
            if len(mini_dict)>0:
                mini_dict['pos_reid']      = [i]
                mini_dict['path']          = [path]
                mini_dict['score_reid']    = [float(v)]
                mini_dict['score_penalty'] = [float(s)]
                out_dict[i]                = mini_dict
        return out_dict



    def preprocess(self, images, is_train):
        def train():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                    split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                    split[i][j] = tf.image.random_flip_left_right(split[i][j])
                    split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                    split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                    split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
                tf.reshape(tf.concat(split[1], axis=0), [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
        def val():
            split = tf.split(images, [1, 1])
            shape = [1 for _ in range(split[0].get_shape()[1])]
            for i in range(len(split)):
                split[i] = tf.reshape(split[i], [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
                split[i] = tf.split(split[i], shape)
                for j in range(len(split[i])):
                    split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                    split[i][j] = tf.image.per_image_standardization(split[i][j])
            return [tf.reshape(tf.concat(split[0], axis=0), [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
                tf.reshape(tf.concat(split[1], axis=0), [self.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
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
    reidentifier = personReIdentifier()
    reidentifier.PersonReIdentification(query_path, gallery_path, save_path, top_k, show_query = False)

