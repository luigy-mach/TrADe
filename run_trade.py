
import numpy  as np
import pandas as pd
import fnmatch
import os
import re

from os.path import splitext as ossplitext
from tqdm    import tqdm

from utils.handler_file import create_dir, copy_files_from_array



def find_files(path_main, pattern, type='separate'):
	list_return = list()
	for dirpath, dirs, files in os.walk(path_main):
		for fname in fnmatch.filter(files, pattern):
			list_return.append((dirpath,fname))

	if type == 'separate':
		list_return = np.asarray(list_return)
		df 			= pd.DataFrame(list_return, columns = ['path','file'])
		df 			= df.sort_values(by=['path'], ascending=True)
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

def find_files_yield(path_main, pattern, type='separate'):
	if type == 'separate':
		list_return = list()
		for dirpath, dirs, files in os.walk(path_main):
			for fname in fnmatch.filter(files, pattern):
				yield (dirpath,fname)
	if type == 'absolute':
		list_return = list()
		for dirpath, dirs, files in os.walk(path_main):
			for fname in fnmatch.filter(files, pattern):
				yield os.path.join(dirpath,fname)
	else:
		return False


def find_dirs(path_main, pattern, type='separate'):
	list_return = list()
	for dirpath, dirs, files in os.walk(path_main):
		for dname in fnmatch.filter(dirs, pattern):
			list_return.append((dirpath,dname))
	if type == 'separate':
		list_return = np.asarray(list_return)
		df 			= pd.DataFrame(list_return, columns = ['path','dir'])
		# df 			= df.sort_values(by=['path'], ascending=True)
		return df.to_numpy()

	if type == 'absolute':
		new_list = list()
		for i,j in list_return:
			new_list.append(os.path.join(i,j))
		# new_list = sorted(new_list )
		return np.asarray(new_list)
	else:
		print('error, you need choise type: [separate,absolute]')
		return None



def find_dirs_yield(path_main, pattern, type='separate'):
	if type == 'separate':
		list_return = list()
		for dirpath, dirs, files in os.walk(path_main):
			for dname in fnmatch.filter(dirs, pattern):
				yield (dirpath,dname)
	if type == 'absolute':
		list_return = list()
		for dirpath, dirs, files in os.walk(path_main):
			for dname in fnmatch.filter(dirs, pattern):
				yield os.path.join(dirpath,dname)
	else:
		return False

def linspace(start, end, step=1.):
    assert start<end, "start need greater than end"
#     epsilon = 0.000001
#     result = np.arange(start,end+epsilon,step)
    result = np.arange(start,end,step)
    result =  np.around(result, decimals=3)
    result = list(result)
    if result[-1]!=end:
        result.append(end)     
    return result




if __name__ == '__main__':


###############################################################
### conda activate trade-py37
###############################################################


	videoPath     = './testing/video'	
	videoFile     = 'video_cam_a_shortcut5.avi'


	max_tracklet        = 20

	videoName, videoExt = os.path.splitext(videoFile)
	packName            = "max-tracklet-{}_{}".format(max_tracklet,videoName)
	savedPath           = os.path.join(videoPath, packName )
	absPathVideo        = os.path.join(videoPath, videoFile)


	###############################################################
	### generate tracklets
	###############################################################


	print("********************************************************************")
	print("executing Generate Tracklets - INIT")

	import time
	from tracklet.interface import generate_tracklets
	
	### tracklet example: 5,10,20,etc
	total_time    = generate_tracklets( absPathVideo, 
				   					 	videoPath, 
				   					 	max_tracklet  = max_tracklet,
				   					 	return_time   = True,
				   					 	packName      = packName,
				   					 	show_window   = False,
				   					 	objDect       = 'yolov3-FFPRID')
		
	print("generated tracklets from videoFile: {} --- {}s seconds ---".format(videoFile, total_time))

	column_names = ['VideoFile', 'Time(sec)']
	df_tmp       = pd.DataFrame([[videoFile, total_time]],columns = column_names)
	csv_path     = os.path.join(videoPath, packName,'time-executing_tracklet_{}.csv'.format(videoName))
	df_tmp.to_csv(csv_path, index=False)

	print("executing Generate Tracklets - END")
	print("********************************************************************")

	##############################################################
	## DOC executing
	##############################################################
	
	print("********************************************************************")
	print("executing One Class Classification - INIT")
	from occ.interface_v2 import OneClassClassifier, create_gallery_inliers

	gallery_path  = os.path.join(savedPath,'cropping')
	
	occ              = OneClassClassifier()	
	time_occ, _      = create_gallery_inliers(occ, gallery_path, savedPath, top_k_inliers=1, return_time=True)
	occ.close_session()
	
	print("executing DOC: --- {}s seconds ---".format( time_occ))

	column_names 	 = ['VideoFile', 'Time(sec)']
	df_tmp       	 = pd.DataFrame([[videoFile, time_occ]], columns = column_names)
	csv_path     	 = os.path.join(savedPath, 'time-executing_occ_{}.csv'.format(videoName))
	df_tmp.to_csv(csv_path, index=False)

	
	print("executing One Class Classification - END")
	print("********************************************************************")
	
###############################################################
###  executing reidentificaton 
###############################################################

	queryMainPath = './testing/query'
	queryPattern  = 'query.png'

	gallery_OCC   = os.path.join(savedPath, 'gallery_inliers_top_1')
	qAbsPaths	  = [os.path.join(queryMainPath,f) for f in os.listdir(queryMainPath) if os.path.isfile(os.path.join(queryMainPath,f))]

	###############################################################
	###  executing reidentificaton with BoT ResNet
	###############################################################


	import time
	from pReID.reid_baseline.interface import Reid_baseline


	print("********************************************************************")
	print("executing Reidentification Bot - INIT")
	for qAbsPath in qAbsPaths:
		qPath, qfile  = os.path.split(qAbsPath) 
		qName, qExt   = os.path.splitext(qfile)
		pathSaveReid  = create_dir(os.path.join(savedPath,'ReId-BoT_results_{}'.format(qName)))


		model                       = Reid_baseline()
		# breakpoint()
		bImgs_path, bDist_mat, time = model.predict(qAbsPath, gallery_OCC, return_time=True)
		copy_files_from_array(bImgs_path, pathSaveReid, extension='*.png')

		c1            = np.asarray([qAbsPath for _ in bImgs_path]).reshape(-1,1)
		c2            = bImgs_path.reshape(-1,1)
		c3            = bDist_mat.reshape(-1,1)
		c4            = time.reshape(-1,1)

		col           = ['queryPath', 'imgsPath', 'cosine_similarity', 'time']
		data          = np.hstack((c1,c2,c3,c4))
		
		df            = pd.DataFrame(data, columns = col)
		baseNameQuery = os.path.basename(queryMainPath)
		dfPathSave    = os.path.join(pathSaveReid,'Bot-dataframe_{}.csv'.format(qName))
		df.to_csv(dfPathSave, index=False, header=not os.path.exists(dfPathSave)) # mode='a'
		
	del bImgs_path
	del bDist_mat
	del time
	del model
	print("executing Reidentification Bot - END")
	print("********************************************************************")

	##############################################################
	##  executing reidentificaton with  SiamIDL 
	##############################################################


	import time
	from pReID.reid_siamDL.interface import Reid_SiamlDL


	print("********************************************************************")
	print("executing Reidentification SiamIDL - INIT")
	model            = Reid_SiamlDL()
	for qAbsPath in qAbsPaths:
		qPath, qfile  = os.path.split(qAbsPath) 
		qName, qExt   = os.path.splitext(qfile)
		pathSaveReid  = create_dir(os.path.join(savedPath,'ReId-SiamIDl_results_{}'.format(qName)))

		bImgs_path, bDist_mat, time = model.predict(qAbsPath, gallery_OCC, return_time=True)
		copy_files_from_array(bImgs_path, pathSaveReid, extension='*.png')

		c1            = np.asarray([qAbsPath for _ in bImgs_path]).reshape(-1,1)
		c2            = bImgs_path.reshape(-1,1)
		c3            = bDist_mat.reshape(-1,1)
		c4            = time.reshape(-1,1)

		col           = ['queryPath', 'imgsPath', 'cosine_similarity', 'time']
		data          = np.hstack((c1,c2,c3,c4))
		
		df            = pd.DataFrame(data, columns = col)
		baseNameQuery = os.path.basename(queryMainPath)
		dfPathSave    = os.path.join(pathSaveReid,'SiamIDL-dataframe_{}.csv'.format(qName))
		df.to_csv(dfPathSave, index=False, header=not os.path.exists(dfPathSave)) # mode='a'
		

	del bImgs_path
	del bDist_mat
	del time
	del model
	print("executing Reidentification SiamIDL - END")
	print("******************************************************************")
