
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import os


from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler

import tensorflow
from tensorflow import keras
from keras.models import load_model



from .functions_luigy import *



os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
print(os.environ["CUDA_VISIBLE_DEVICES"])


from numba import cuda 
device = cuda.get_current_device()
device.reset()


gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tensorflow.config.experimental.set_memory_growth(gpu, True)

	except RuntimeError as e:
		print(e)

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

	# for idx_dir in tqdm(dir_gallery):
	for idx_dir in list_dir_yield(path_gallery):
		gpath                             = os.path.join(path_gallery, idx_dir)
		_, best_images_path_shorted, _, _ = occ.generate_inliers_outliers(gpath)
		pick_inliers                      = min(top_k_inliers, len(best_images_path_shorted))

		for i in range(pick_inliers):
			new_path = add_sth_path(best_images_path_shorted[i], i, dst_path=save_path ,prefix='inliers')
			copy_file(best_images_path_shorted[i], new_path)
		best_images_path_shorted = None
		pick_inliers             = None

	time_occ_end = time.time()

	if return_time:
		return time_occ_end-time_occ_init
	else:
		return


class OneClassClassifier():
	
	def __init__(self, save_path=None, shape_img=(224,224,3)):
		self.path_model_t = '/home/luigy/luigy/develop/re3/tracking/occ/model/train/model_t_smd_33epoch.h5'
		self.s_path       = '/home/luigy/luigy/develop/re3/tracking/occ/model/pkl_files'
		self.ms_file      = 'MinMaxScaler_fit_train_normal-person.pkl'
		self.clf_file     = 'LocalOutlierFactor_fit_train_normal-person.pkl'

		self.shape_img 	  = shape_img

		self._model       = keras.models.load_model(self.path_model_t, compile = False)

		# data_path                     = '/home/luigy/luigy/datasets/OCC_one_class/OCC_prepare_VOC2012_224x224'
		# x_train_normal, _, _, _, _, _ = open_mydata(data_path)
		# train_normal                  = model.predict(x_train_normal)
		# train_normal                  = train_normal.reshape((len(x_train_normal),-1))

		# ##Convert to 0-1
		# ms                            = MinMaxScaler().fit(train_normal)
		# norm_train_normal             = ms.transform(train_normal)
		# clf                           = LocalOutlierFactor(n_neighbors = 20, novelty = True, contamination = 0.1).fit(norm_train_normal)

		# save_model(ms,  os.path.join(s_path, ms_file) )
		# save_model(clf, os.path.join(s_path, clf_file) )

		self._ms2       = load_model(  os.path.join(self.s_path, self.ms_file) )
		self._clf2      = load_model(  os.path.join(self.s_path, self.clf_file) )
		self._save_path = save_path
		if save_path is None:
			self._save_path = create_dir('./save_path_temp')


	def set_dir_save(self, path):
		self._save_path = path
		return os.path.exists(self._save_path)


	def predict_occ_features(self, images):
		predict_feats = self._model.predict(images)
		predict_feats = predict_feats.reshape((len(images),-1))
		return predict_feats


	def predict_InOut_liers(self, predict_feats):
		norm_predict_feats = self._ms2.transform(predict_feats)
		inliers            = self._clf2.predict(norm_predict_feats)
		query_inliers      = np.where(inliers<0)
		query_outliers     = np.where(inliers>0)
		return query_inliers[0], query_outliers[0]


	def predict_occ(self, images):
		predict_feats                 = self.predict_occ_features(images)
		query_inliers, query_outliers = self.predict_InOut_liers(predict_feats)
		return predict_feats, query_inliers, query_outliers



	
	# def create_gallery_inliers(self, path_gallery, save_path, top_k_inliers=1, return_time=False):
	# 	assert top_k_inliers>0, 'error, top_k_inliers is {}'.format(top_k_inliers)
	# 	dir_gallery = sorted(os.listdir(path_gallery))
		
	# 	save_path   = os.path.join(save_path,'gallery_inliers_top_{}'.format(top_k_inliers))
	# 	create_dir(save_path)

	# 	import time
	# 	time_occ_init = time.time()

	# 	for idx_dir in tqdm(dir_gallery):
	# 		gpath                             = os.path.join(path_gallery, idx_dir)
	# 		_, best_images_path_shorted, _, _ = self.generate_inliers_outliers(gpath)
	# 		pick_inliers                      = min(top_k_inliers, len(best_images_path_shorted))

	# 		for i in range(pick_inliers):
	# 			new_path = add_sth_path(best_images_path_shorted[i], i, dst_path=save_path ,prefix='inliers')
	# 			copy_file(best_images_path_shorted[i], new_path)
	# 		best_images_path_shorted = None
	# 		pick_inliers             = None

	# 	time_occ_end = time.time()

	# 	if return_time:
	# 		return time_occ_end-time_occ_init
	# 	else:
	# 		return


	def generate_inliers_outliers(self, path_gallery, save_path=None):
		images, path_images                          = openImages_FromDirectory(path_gallery, shape = self.shape_img, normalize = True)
		path_images                                  = np.asarray(path_images)
		predict_feats, query_inliers, query_outliers = self.predict_occ(images)

		images_inliers, path_images_inliers          = apply_query_ON_array(query_inliers,  images, path_images)
		worst_images,   worst_images_path_shorted    = apply_query_ON_array(query_outliers, images, path_images)

		predict_feats_inliers                        = predict_feats[query_inliers]		
		_, indices_sorted, _                         = sorted_predicts(predict_feats_inliers)
		best_images, best_images_path_shorted        = apply_query_ON_array(indices_sorted, images_inliers, path_images_inliers )
		

		if save_path is not None:
			path_in  = create_dir(os.path.join(save_path,'inliers' ))
			path_out = create_dir(os.path.join(save_path,'outliers'))

			for i, item in enumerate(best_images_path_shorted):
				new_path = add_sth_path(item, i, dst_path=path_in ,prefix='inliers')
				copy_file(item, new_path)

			for i, item in enumerate(worst_images_path_shorted):
				new_path = add_sth_path(item, i, dst_path=path_out ,prefix='outliers')
				copy_file(item, new_path)

		images                = None		
		predict_feats         = None
		query_inliers         = None
		query_outliers        = None
		predict_feats_inliers = None
		indices_sorted        = None

		return best_images, best_images_path_shorted, worst_images, worst_images_path_shorted  		
		

	def generate_sheet_inliers_outliers(self, path_gallery, idx_dir=None ):
		images, path_images   = openImages_FromDirectory(path_gallery, idx_dir, shape=shape_img, normalize=True)
		path_images           = np.asarray(path_images)
		images_labels         = np.asarray([i for i in range(len(images))])
		try: 
			predict, query_inliers, query_outliers = self.predict_occ(images)

			num_inliers_detec  = len(query_inliers)
			num_outliers_detec = len(query_outliers)

			predict_inliers                                              = predict[query_inliers]
			images_inliers, images_labels_inliers, path_images_inliers   = apply_query_ON_array(query_inliers, images, images_labels, path_images)
			worst_images, worst_images_labels, worst_images_path_shorted = apply_query_ON_array(query_outliers, images, images_labels, path_images)
			array_distances_sorted, array_for_sorted, mean               = sorted_predicts(predict_inliers)
			best_images, best_images_labels, best_images_path_shorted    = apply_query_ON_array(array_for_sorted, images_inliers, images_labels_inliers, path_images_inliers )
			
			caption_msg       = None
			#		caption_msg       = '{} inliers were detected.'
			#		caption_msg       = caption_msg.format(num_inliers_detec)
			show_imagesGrid( best_images, 
							best_images_labels, 
							array_distances       = array_distances_sorted,
							images_array_outliers = worst_images,
							labels_outliers       = worst_images_labels,
							caption_msg           = caption_msg ,
							key                   = idx_dir ,
							# savePath              = save_path_processing,
							savePath              = self._save_path,
							saveName              = '{}.jpg'.format(idx_dir),
							imshow                = False,
							)

			if  num_inliers_detec > 2: ## 3 o mas inliers
				# array_result[idx_dir] = [len(images),num_inliers_detec, num_outliers_detec, True]            
				return idx_dir, [len(images),num_inliers_detec, num_outliers_detec, True]            
			else:
				# array_result[idx_dir] = [len(images), num_inliers_detec, num_outliers_detec, False]
				return idx_dir, [len(images), num_inliers_detec, num_outliers_detec, False]
		except:
			print("error: ", idx_dir)
			# array_result[idx_dir] = [len(images),-1,-1,'error']
			return idx_dir,  [len(images),-1,-1,'error']


if __name__ == '__main__':

    
    # ### Itereate GALLERY generate video TownCentreXVID

	gallery_path   = '/home/luigy/luigy/develop/re3/tracking/resultados/pack_5_min_complete_v3/cropping/sample_TownCentreXVID'
	# gallery_path   = '/home/luigy/luigy/develop/re3/tracking/resultados/pack_5_min_complete_v3/cropping/TownCentreXVID'
	save_path_main = '/home/luigy/luigy/develop/re3/tracking/occ/results'


	occ = OneClassClassifier()

	occ.create_gallery_inliers(gallery_path, save_path_main, top_k_inliers=1)
	# for i in range(1,21):
	# 	occ.create_gallery_inliers(gallery_path, save_path_main, top_k_inliers=i)


