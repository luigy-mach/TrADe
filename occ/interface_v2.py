
import numpy as np
import os
import gc

# from tensorflow import keras
import tensorflow
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

from .functions_luigy_v2 import *
from tqdm import tqdm



# import gc
# tensorflow.reset_default_graph()
# tensorflow.keras.backend.clear_session()
# gc.collect()

# os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
# print(os.environ["CUDA_VISIBLE_DEVICES"])


# # from numba import cuda 
# # device = cuda.get_current_device()
# # device.reset()

# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
# cuda.select_device(0)
# cuda.close()


# gpus = tensorflow.config.experimental.list_physical_devices('GPU')
# if gpus:
# 	try:
# 		for gpu in gpus:
# 			tensorflow.config.experimental.set_memory_growth(gpu, True)
# 			print("gpu <----------------------------------------------------------------")


# 	except RuntimeError as e:
# 		print("ERROR <---------------------------------------------------------------")
# 		print(e)



def list_dir_yield(path):
	dir_gallery = sorted(os.listdir(path))
	for i in dir_gallery:
		yield i
	
def create_gallery_inliers(occ, path_gallery, save_path, top_k_inliers=1, return_time=False):
	assert top_k_inliers>0, 'error, top_k_inliers is {}'.format(top_k_inliers)
	dir_gallery = sorted(os.listdir(path_gallery))
	
	save_path   = os.path.join(save_path,'gallery_inliers_top_{}'.format(top_k_inliers))
	create_dir(save_path)

	import time
	time_occ_init = time.time()

	time_predict = 0

	# for idx_dir in tqdm(dir_gallery):
	for idx_dir in list_dir_yield(path_gallery):
		gpath                        = os.path.join(path_gallery, idx_dir)
		temp_test                    = time.time()
		_, path_images, _            = occ.generate_inliers_outliers(gpath)
		temp_test                    = time.time() - temp_test
		time_predict                 = time_predict + temp_test                 
		
		pick_inliers                 = min(top_k_inliers, len(path_images))

		for i in range(pick_inliers):
			new_path = add_sth_path(path_images[i], i, dst_path=save_path ,prefix='inliers')
			copy_file(path_images[i], new_path)
		path_images  = None
		pick_inliers = None

	time_occ_end = time.time()

	if return_time:
		return time_occ_end-time_occ_init , time_predict
	else:
		return None


class OneClassClassifier():
	
	def __init__(self, save_path=None, shape_img=(224,224,3)):

		self.shape_img 	     = shape_img

		self.path_model_t    = "./occ/model/train_v2/model_t_smd_149epoch.h5"
		self.path_load_train = "./occ/model/pkl_files_v2/predictTrain.npy"

		self._model          = tensorflow.keras.models.load_model(self.path_model_t, compile = False)
		self._predictTrain   = load_npy( self.path_load_train )
		
		self._ms2            = None
		self._ms2            = MinMaxScaler()
		self._predictTrain   = self._ms2.fit_transform(self._predictTrain)

		self._clf2           = LocalOutlierFactor(n_neighbors=5, novelty=True)
		self._clf2.fit(self._predictTrain)


		self._save_path = save_path
		if save_path is None:
			self._save_path = create_dir('./save_path_temp')


	def predict_occ_features(self, images):
		predict_feats = self._model.predict(images)
		predict_feats = predict_feats.reshape((len(images),-1))
		return predict_feats


	def predict_InOut_liers(self, predict_feats, round_dec=3):
		norm_predict_feats =  self._ms2.transform(predict_feats)
		scores             = -self._clf2.decision_function(norm_predict_feats)
		idx_scorex         = np.argsort(scores)
		scores             = np.round(scores,round_dec)
		return idx_scorex, scores[idx_scorex]



	def predict_occ(self, images):
		predict_feats             = self.predict_occ_features(images)
		idx_sorted, sorted_scores = self.predict_InOut_liers(predict_feats)
		return idx_sorted, sorted_scores



	def generate_inliers_outliers(self, path_gallery, save_path=None):
		images, path_images  = openImages_FromDirectory(path_gallery, shape = self.shape_img, normalize = True)
		path_images          = np.asarray(path_images)

		idx_sorted, sorted_scores = self.predict_occ(images)

		images               = images[idx_sorted]
		path_images          = path_images[idx_sorted]


		if save_path is not None:
			path_in  = create_dir(os.path.join(save_path,'imgs_sorted' ))

			for i, item in enumerate(images):
				new_path = add_sth_path(item, i, dst_path=path_in, prefix='inliers')
				copy_file(item, new_path)

		return images, path_images, sorted_scores 		
		

	def generate_sheet_inliers_outliers(self, path_gallery, idx_dir=None ):
		images, path_images   = openImages_FromDirectory(path_gallery, idx_dir, shape=shape_img, normalize=True)
		path_images           = np.asarray(path_images)
		images_labels         = np.asarray([i for i in range(len(images))])
		try: 
			idx_sorted, sorted_scores   = self.predict_occ(images)
		
			images      = images[idx_sorted]
			path_images = path_images[idx_sorted]

			
			caption_msg       = None
			#		caption_msg       = '{} inliers were detected.'
			#		caption_msg       = caption_msg.format(num_inliers_detec)
			show_imagesGrid( images, 
							 sorted_scores, 
							array_distances       = sorted_scores,

							caption_msg           = caption_msg ,
							key                   = idx_dir ,
							# savePath              = save_path_processing,
							savePath              = self._save_path,
							saveName              = '{}.jpg'.format(idx_dir),
							imshow                = False,
							)

			return idx_dir, [len(images)]

		except:
			print("error: ", idx_dir)
			return idx_dir, [len(images),'error']



	def close_session(self):
		tensorflow.keras.backend.clear_session() 
		gc.collect()
		return 



if __name__ == '__main__':

    
    # ### Itereate GALLERY generate video

	gallery_path   = '/home/luigy/luigy/develop/re3/tracking/dataset_prid2011/raw_video_yolo+re3/cam_a/dir_max_tracklet_20_cam_a.avi_ok_v1/cropping_test'
	save_path_main = '/home/luigy/luigy/develop/re3/tracking/occ/results'

	occ          = OneClassClassifier()
	time1, time2 = create_gallery_inliers(occ, gallery_path, save_path_main, top_k_inliers=1, return_time=True)

	print("time : ", time1, time2 )