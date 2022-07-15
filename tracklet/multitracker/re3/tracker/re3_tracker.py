import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import time

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from tracker import network

from re3_utils.util import bb_util
from re3_utils.util import im_util
from re3_utils.tensorflow_util import tf_util

# Network Constants
from constants import CROP_SIZE
from constants import CROP_PAD
from constants import LSTM_SIZE
from constants import LOG_DIR
from constants import GPU_ID
from constants import MAX_TRACK_LENGTH
from termcolor import colored

# SPEED_OUTPUT = True
SPEED_OUTPUT = False

class Re3Tracker(object):
    def __init__(self, gpu_id=GPU_ID):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        basedir                            = os.path.dirname(__file__)
        self.g1                            = tf.Graph()
        self.sess                          = tf_util.Session(name_graph = self.g1)
        with self.g1.as_default():
            self.imagePlaceholder                  = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
            self.prevLstmState                     = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
            self.batch_size                        = tf.placeholder(tf.int32, shape=())
            self.outputs, self.state1, self.state2 = network.inference(
                    self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                    prevLstmState=self.prevLstmState)
            #self.sess = tf_util.Session()
            # self.sess = tf_util.Session(name_graph=self.g1)
            self.sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(os.path.join(basedir, '..', LOG_DIR, 'checkpoints'))
            if ckpt is None:
                raise IOError(
                        ('Checkpoint model could not be found. '
                        'Did you download the pretrained weights? '
                        'Download them here: http://bit.ly/2L5deYF and read the Model section of the Readme.'))
            tf_util.restore(self.sess, ckpt.model_checkpoint_path)

            self.tracked_data        = {}
            self.time                = 0
            self.total_forward_count = -1


    # unique_id{str}: A unique id for the object being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_box{None or 4x1 numpy array or list}: 4x1 bounding box in X1, Y1, X2, Y2 format.
    def track(self, unique_id, image, starting_box=None):
        start_time = time.time()

        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()

        image_read_time = time.time() - start_time

        if starting_box is not None:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = np.array(starting_box) # turns list into numpy array if not and copies for safety.
            prevImage = image
            originalFeatures = None
            forwardCount = 0
        elif unique_id in self.tracked_data:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)

        feed_dict = {
                self.imagePlaceholder : [croppedInput0, croppedInput1],
                self.prevLstmState : lstmState,
                self.batch_size : 1,
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        lstmState = [s1[0], s1[1], s2[0], s2[1]]
        if forwardCount == 0:
            originalFeatures = [s1[0], s1[1], s2[0], s2[1]]

        prevImage = image

        # Shift output box to full image coordinate system.
        outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)

        if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
            croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
            input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
            feed_dict = {
                    self.imagePlaceholder : input,
                    self.prevLstmState : originalFeatures,
                    self.batch_size : 1,
                    }
            rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
            lstmState = [s1[0], s1[1], s2[0], s2[1]]

        forwardCount += 1
        self.total_forward_count += 1

        if starting_box is not None:
            # Use label if it's given
            outputBox = np.array(starting_box)

        self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed:   %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed: %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed:      %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBox


    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    { unique_id: {X1, Y1, X2, Y2} }
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, unique_ids, image, starting_boxes=None):
        start_time = time.time()
        #Error type of unique_ids must be list 
        assert type(unique_ids) == list, 'unique_ids must be a list for multi_track'
        #Errors length of unique_ids greater than 1  
        assert len(unique_ids) > 1, 'unique_ids must be at least 2 elements'


        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()
        
        height_image, width_image = image.shape[:2]

        image_read_time = time.time() - start_time

        # Get inputs for each track.
        images = []
        lstmStates = [ [] for _ in range(4) ]
        pastBBoxesPadded = []

        if starting_boxes is None:
            starting_boxes = dict()

        for unique_id in unique_ids:
            if unique_id in starting_boxes:
                lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox = np.array(starting_boxes[unique_id]) # turns list into numpy array if not and copies for safety.
                prevImage = image
                originalFeatures = None
                forwardCount = 0
                self.tracked_data[unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
            elif unique_id in self.tracked_data:
                lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            else:
                raise Exception('Unique_id %s with no initial bounding box' % unique_id)

            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded.append(pastBBoxPadded)
            images.extend([croppedInput0, croppedInput1])
            for ss,state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        feed_dict = {
                self.imagePlaceholder : images,
                self.prevLstmState : lstmStateArrays,
                self.batch_size : len(images) / 2
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        outputBoxes = np.zeros((len(unique_ids), 4))

        test_output=dict()
        for uu,unique_id in enumerate(unique_ids):
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

            prevImage = image

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
                croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
                input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
                feed_dict = {
                        self.imagePlaceholder : input,
                        self.prevLstmState : originalFeatures,
                        self.batch_size : 1,
                        }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
                lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            forwardCount += 1
            self.total_forward_count += 1

            if unique_id in starting_boxes:
                # Use label if it's given
                outputBox = np.array(starting_boxes[unique_id])

            outputBoxes[uu,:] = outputBox
            test_output[unique_id] = outputBox.tolist()
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
        
            


        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed per object: %.3f FPS' % (len(unique_ids) / (end_time - start_time - image_read_time)))
            print('Current tracking speed per frame:  %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed:          %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed per object:    %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBoxes, test_output
        #return outputBoxes

   

    def multi_track_v2(self, unique_ids, image, starting_boxes=None):
        start_time = time.time()
        #Error type of unique_ids must be list 
        #assert type(unique_ids) == list, 'unique_ids must be a list for multi_track'
        #Errors length of unique_ids greater than 1  
        #assert len(unique_ids) > 1, 'unique_ids must be at least 2 elements'


        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()
        
        error_bound = 10 #error pixels bb predict bound limits
        height_image, width_image = image.shape[:2]

        image_read_time = time.time() - start_time

        # Get inputs for each track.
        images = []
        lstmStates = [ [] for _ in range(4) ]
        pastBBoxesPadded = []

        if starting_boxes is None:
            starting_boxes = dict()

        for unique_id in unique_ids:
            if unique_id in starting_boxes:
                lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox = np.array(starting_boxes[unique_id]) # turns list into numpy array if not and copies for safety.
                prevImage = image
                originalFeatures = None
                forwardCount = 0
                self.tracked_data[unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
            elif unique_id in self.tracked_data:
                lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            else:
                raise Exception('Unique_id %s with no initial bounding box' % unique_id)

            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded.append(pastBBoxPadded)
            images.extend([croppedInput0, croppedInput1])
            for ss,state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        feed_dict = {
                self.imagePlaceholder : images,
                self.prevLstmState : lstmStateArrays,
                self.batch_size : len(images) / 2
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        
        outputBoxes = np.zeros((len(unique_ids), 4)) # solo salida, no importa
        test_output=dict()
        new_unique_ids = unique_ids.copy()

        for uu,unique_id in enumerate(unique_ids):
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

            prevImage = image

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
                croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
                input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
                feed_dict = {
                        self.imagePlaceholder : input,
                        self.prevLstmState : originalFeatures,
                        self.batch_size : 1,
                        }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
                lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            forwardCount += 1
            self.total_forward_count += 1


            #if (int(outputBox[0])>0 and int(outputBox[1])>0) and (int(outputBox[2])<width_image-error_bound and int(outputBox[3])<height_image-error_bound):
            if (int(outputBox[0])<0+error_bound or int(outputBox[1])<0+error_bound) or (int(outputBox[2])>width_image-error_bound or int(outputBox[3])>height_image-error_bound):
                print("borrandoooooooo1")
                print("borrrareeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ", unique_id)
                print(len(self.tracked_data))
                del self.tracked_data[unique_id]
                try:
                    del starting_boxes[unique_id]
                except:
                    print("don't find element")
                print(len(self.tracked_data))
                new_unique_ids.remove(unique_id)
                print("borrandoooooooo2")
                print("new tam ids")
                print(len(new_unique_ids))
                print("new tam tracked_data")
                print(len(self.tracked_data))
                print("borrandoooooooo4")
                if len(self.tracked_data)==0:
                    return outputBoxes, test_output, self.tracked_data
                print(self.tracked_data.keys())    
                print("borrandoooooooo5fin")

            # solo si inserto nuevos starting_boxes actualizo nueva posicion 
            if unique_id in starting_boxes:
                # Use label if it's given
                outputBox = np.array(starting_boxes[unique_id])

            outputBoxes[uu,:] = outputBox # esta demas
            test_output[unique_id] = outputBox.tolist()# esta demas
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
            


        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed per object: %.3f FPS' % (len(unique_ids) / (end_time - start_time - image_read_time)))
            print('Current tracking speed per frame:  %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed:          %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed per object:    %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBoxes, test_output, self.tracked_data
        #return outputBoxes

    

    def clear_multi_tracking(self):
        self.tracked_data.clear()

    
    def multi_track_v3(self, unique_ids, image, starting_boxes=None):
        # with self.g1.as_default():

        start_time = time.time()
        #Error type of unique_ids must be list 
        assert type(unique_ids) == list, 'unique_ids must be a list for multi_track'
        #Errors length of unique_ids greater than 1  
        assert len(unique_ids) > 1, 'unique_ids must be at least 2 elements'



        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()
        
        error_bound = 0 #error pixels bb predict bound limits
        
        height_image, width_image = image.shape[:2]

        image_read_time = time.time() - start_time

        # Get inputs for each track.
        images           = []
        lstmStates       = [ [] for _ in range(4) ]
        pastBBoxesPadded = []
        ids_delete       = []

        if starting_boxes is None:
            # print(" no hay starting_boxes")
            starting_boxes = dict()
            if len(self.tracked_data) != len(unique_ids):
                unique_ids = set(unique_ids)
                tmp        = set(list(self.tracked_data.keys()))
                ids_delete = tmp.difference(unique_ids)
                if len(ids_delete)>0:
                    for x in ids_delete:
                        del self.tracked_data[x]

                unique_ids = list(self.tracked_data.keys())

        # print("entrante unique_ids")
        # print(unique_ids)
        # print("actual self.tracked_data.keys()")
        # print([key for key in self.tracked_data.keys()])


        for unique_id in unique_ids:
            if unique_id in starting_boxes:
                lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox = np.array(starting_boxes[unique_id]) # turns list into numpy array if not and copies for safety.
                prevImage = image
                originalFeatures = None
                forwardCount = 0
                self.tracked_data[unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
            elif unique_id in self.tracked_data:
                lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            else:
                raise Exception('Unique_id %s with no initial bounding box' % unique_id)

            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded.append(pastBBoxPadded)
            images.extend([croppedInput0, croppedInput1])
            for ss,state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        feed_dict = {
                self.imagePlaceholder : images,
                self.prevLstmState : lstmStateArrays,
                self.batch_size : len(images) / 2
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        outputBoxes = np.zeros((len(unique_ids), 4))

        outputBoxes = np.zeros((len(unique_ids), 4)) # solo salida, no importa
        test_output=dict()
        new_unique_ids = unique_ids.copy()

        for uu,unique_id in enumerate(unique_ids):
            # print(colored(unique_ids,'magenta'))
            # print(colored(uu,'magenta'))
            # print(colored(unique_id,'magenta'))
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

            prevImage = image

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
                croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
                input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
                feed_dict = {
                        self.imagePlaceholder : input,
                        self.prevLstmState : originalFeatures,
                        self.batch_size : 1,
                        }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
                lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            forwardCount += 1
            self.total_forward_count += 1


            #if (int(outputBox[0])>0 and int(outputBox[1])>0) and (int(outputBox[2])<width_image-error_bound and int(outputBox[3])<height_image-error_bound):
            # if (int(outputBox[0])<0+error_bound or int(outputBox[1])<0+error_bound) or (int(outputBox[2])>width_image-error_bound or int(outputBox[3])>height_image-error_bound):
            x0 = int(outputBox[0])
            y0 = int(outputBox[1])
            x1 = int(outputBox[2])
            y1 = int(outputBox[3])



            if x0>x1:
                # while(1):
                    # print("cambieeeeeeeeeeee x0")
                tmp=x0
                x0=x1
                x1=tmp
            if y0>y1:
                # while(1):
                    # print("cambieeeeeeeeeeee y0")
                tmp=y0
                y0=y1
                y1=y0
                
            xx0 = 0+error_bound
            yy0 = 0+error_bound
            xx1 = width_image-error_bound
            yy1 = height_image-error_bound
            # if int(outputBox[0])<0+error_bound or int(outputBox[1])<0+error_bound or int(outputBox[2])>width_image-error_bound or int(outputBox[3])>height_image-error_bound:
            # if x0<0+error_bound or y0<0+error_bound or x1>width_image-error_bound or y1>height_image-error_bound:
            
            test_output[unique_id] = outputBox.tolist() #no sirve
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)

            if x0<xx0 or y0<yy0 or x1>xx1 or y1>yy1:
                # print(colored("borrandoooooooo INIT",'red'))
                # print(colored(outputBox,'blue'))                
                # print("se borrara:  ", unique_id)
                # print("tamano tracked_data: ",len(self.tracked_data))
                # print([int(key) for key in self.tracked_data.keys()])
                # # del self.tracked_data[unique_id]
                # print(colored(self.tracked_data.keys(),'green'))
                try:
                    # print(colored("find element and remove",'yellow'))
                    del self.tracked_data[unique_id]
                except:
                    print(colored("don't find element",'yellow'))
                if unique_id in self.tracked_data:
                    del self.tracked_data[unique_id]

                # print("nuevo tamano tracked_data: ", len(self.tracked_data))
                # print([int(key) for key in self.tracked_data.keys()])
                # print("tamano unique_ids:" ,len(unique_ids))
                # # print("tamano unique_ids:" ,len(new_unique_ids))
                # # unique_ids.remove(unique_id)
                # print(colored(type(unique_ids),'green'))
                # print(colored(type(self.tracked_data),'green'))
                # # new_unique_ids.remove(unique_id)
                # print("nuevo tamano unique_ids:" ,len(unique_ids))
                # # print("nuevo tamano unique_ids:" ,len(new_unique_ids))

                # if len(self.tracked_data)==0:
                    ##error arreglar
                    # return outputBoxes, test_output, self.tracked_data
                    # print(colored("borrandoooooooo END   if len(self.tracked_data)==0:",'red'))
                    # return outputBoxes, test_output, self.tracked_data , unique_ids

                # print(self.tracked_data.keys())    
                # print(colored("borrandoooooooo END",'red'))
                if unique_id in test_output:
                    del test_output[unique_id] #no sirve
            # else:
                # print(colored("********************************************************",'cyan'))
            
                # if unique_id in starting_boxes: # solo se ejecuta cuando ingresa nuevo starting_boxes
                    # Use label if it's given
                    # outputBox = np.array(starting_boxes[unique_id])

                # outputBoxes[uu,:] = outputBox #no sirve
                # test_output[unique_id] = outputBox.tolist() #no sirve
                # self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)


            # if unique_id in starting_boxes: # solo se ejecuta cuando ingresa nuevo starting_boxes
                # Use label if it's given
                # outputBox = np.array(starting_boxes[unique_id])
            
            # outputBoxes[uu,:] = outputBox #no sirve
            # test_output[unique_id] = outputBox.tolist() #no sirve
            # self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)


        ids_update = [key for key in self.tracked_data.keys()]
        # unique_ids = new_unique_ids
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed per object: %.3f FPS' % (len(unique_ids) / (end_time - start_time - image_read_time)))
            print('Current tracking speed per frame:  %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed:          %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed per object:    %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        # print(colored("devolviendo :",'red'))
        # print(colored(len(ids_update),'red'))
        # print(colored(ids_update,'red'))
        # print(colored("/////////////////////////////",'red'))
        
        return test_output, test_output, self.tracked_data , ids_update
        # return outputBoxes, test_output, self.tracked_data , ids_update
        #return outputBoxes



    def multi_track_v4(self, unique_ids, image, starting_boxes=None):
        # with self.g1.as_default():

        start_time = time.time()
        #Error type of unique_ids must be list 
        assert type(unique_ids) == list, 'unique_ids must be a list for multi_track'
        #Errors length of unique_ids greater than 1  
        assert len(unique_ids) > 0, 'unique_ids must be at least 2 elements'



        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()
        
        error_bound = 0 #error pixels bb predict bound limits
        
        height_image, width_image = image.shape[:2]

        image_read_time = time.time() - start_time

        # Get inputs for each track.
        images           = []
        lstmStates       = [ [] for _ in range(4) ]
        pastBBoxesPadded = []
        ids_delete       = []

        if starting_boxes is None:
            # print(" no hay starting_boxes")
            starting_boxes = dict()
            if len(self.tracked_data) != len(unique_ids):
                unique_ids = set(unique_ids)
                tmp        = set(list(self.tracked_data.keys()))
                ids_delete = tmp.difference(unique_ids)
                if len(ids_delete)>0:
                    for x in ids_delete:
                        del self.tracked_data[x]

                unique_ids = list(self.tracked_data.keys())

        # print("entrante unique_ids")
        # print(unique_ids)
        # print("actual self.tracked_data.keys()")
        # print([key for key in self.tracked_data.keys()])


        for unique_id in unique_ids:
            if unique_id in starting_boxes:
                lstmState                    = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox                     = np.array(starting_boxes[unique_id]) # turns list into numpy array if not and copies for safety.
                prevImage                    = image
                originalFeatures             = None
                forwardCount                 = 0
                self.tracked_data[unique_id] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
            elif unique_id in self.tracked_data:
                lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            else:
                # breakpoint()
                raise Exception('Unique_id %s with no initial bounding box' % unique_id)

            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1, _              = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded.append(pastBBoxPadded)
            images.extend([croppedInput0, croppedInput1])
            for ss,state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        feed_dict = {
                self.imagePlaceholder : images,
                self.prevLstmState : lstmStateArrays,
                self.batch_size : len(images) / 2
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        # outputBoxes = np.zeros((len(unique_ids), 4))

        # outputBoxes = np.zeros((len(unique_ids), 4)) # solo salida, no importa
        test_output = dict()
        # new_unique_ids = unique_ids.copy()

        for uu,unique_id in enumerate(unique_ids):
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

            prevImage      = image

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox      = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
                croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
                input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
                feed_dict = {
                        self.imagePlaceholder : input,
                        self.prevLstmState : originalFeatures,
                        self.batch_size : 1,
                        }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict = feed_dict)
                lstmState         = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            forwardCount             += 1
            self.total_forward_count += 1

            x0 = int(outputBox[0])
            y0 = int(outputBox[1])
            x1 = int(outputBox[2])
            y1 = int(outputBox[3])

            if x0>x1:
                # while(1):
                    # print("cambieeeeeeeeeeee x0")
                tmp = x0
                x0  = x1
                x1  = tmp
            if y0>y1:
                # while(1):
                    # print("cambieeeeeeeeeeee y0")
                tmp = y0
                y0  = y1
                y1  = y0
                
            xx0 = 0+error_bound
            yy0 = 0+error_bound
            xx1 = width_image-error_bound
            yy1 = height_image-error_bound
       
            test_output[unique_id]       = outputBox.tolist() #no sirve
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)

            if x0<xx0 or y0<yy0 or x1>xx1 or y1>yy1:
                try:
                    # print(colored("find element and remove",'yellow'))
                    del self.tracked_data[unique_id]
                except:
                    print(colored("don't find element",'yellow'))
                if unique_id in self.tracked_data:
                    del self.tracked_data[unique_id]

                if unique_id in test_output:
                    del test_output[unique_id] #no sirve

        # ids_update = [key for key in self.tracked_data.keys()]
        # unique_ids = new_unique_ids
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        # if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
        #     print('Current tracking speed per object: %.3f FPS' % (len(unique_ids) / (end_time - start_time - image_read_time)))
        #     print('Current tracking speed per frame:  %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
        #     print('Current image read speed:          %.3f FPS' % (1 / (image_read_time)))
        #     print('Mean tracking speed per object:    %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        
        return test_output




    


class CopiedRe3Tracker(Re3Tracker):
    def __init__(self, sess, copy_vars, gpu=None):
        self.sess = sess
        self.imagePlaceholder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        network_scope = 'test_network'
        if gpu is not None:
            with tf.device('/gpu:' + str(gpu)):
                with tf.variable_scope(network_scope):
                    self.outputs, self.state1, self.state2 = network.inference(
                            self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                            prevLstmState=self.prevLstmState)
        else:
            with tf.variable_scope(network_scope):
                self.outputs, self.state1, self.state2 = network.inference(
                        self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                        prevLstmState=self.prevLstmState)
        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=network_scope)
        self.sync_op = self.sync_from(copy_vars, local_vars)

        self.tracked_data = {}

        self.time = 0
        self.total_forward_count = -1

    def reset(self):
        self.tracked_data = {}
        self.sess.run(self.sync_op)

    def sync_from(self, src_vars, dst_vars):
        sync_ops = []
        with tf.name_scope('Sync'):
            for(src_var, dst_var) in zip(src_vars, dst_vars):
                sync_op = tf.assign(dst_var, src_var)
                sync_ops.append(sync_op)
        return tf.group(*sync_ops)


