
import numpy  as np
import pandas as pd
import fnmatch
import os
import re

from os.path import splitext as ossplitext
from tqdm    import tqdm

from utils.handler_file import create_dir

# from pyqt5_window.interface    import main_evaluation
from pyqt5_window.interface_v2 import main_evaluation
from utils.handler_file import *


def create_tau_segments_of_video(pathDatasetSkip, pattern_videDatasetSkip, tau_list):

	from utils.dataset_prid2011.videoPRID import split_tauSequence

	files                    = find_files(pathDatasetSkip, pattern_videDatasetSkip, type='separate')
	for tau in tau_list:
		print('Tau  = {}'.format(tau))
		# for i in tqdm(range(len(files))):
		for i, (videoPath, videoFile) in tqdm(enumerate(files)):
			fname, fext    = os.path.splitext(videoFile)
			video_path     = os.path.join(videoPath, videoFile)
			gts_dataframe  = os.path.join(videoPath, 'gts_{}.csv'.format(fname))
			save_path      = os.path.join(videoPath, 'tau_{:05d}'.format(tau))
			split_tauSequence(video_path, gts_dataframe, tau, save_path, show_video=False)



def generate_gallery_to_evaluation(pathDatasetTrade, pattern_Tau, list_max_tracklet, objDect='yolov3-FFPRID'):
	print("********************************************************************")
	print("executing Generate gallery with YOLO - INIT")
	import time
	from tracklet.interface import generate_gallery_yolo
	
	files                 = find_files(pathDatasetTrade, pattern_Tau, type='separate')

	for max_track in list_max_tracklet:
		for i, (videpPath, videoFile) in enumerate(files):
			videoName, videoExt = os.path.splitext(videoFile)
			absPathVideo        = os.path.join(videpPath, videoFile)
			saveResults         = videpPath
			packName            = "max_tracklet_{}_{}".format(max_track, videoName)


			total_time    = generate_gallery_yolo(  absPathVideo, 
													saveResults, 
													max_tracklet  = max_track,
													return_time   = True,
													packName      = packName,
													show_window   = False, 
													objDect       = objDect) #  yolov3-FFPRID or yolov3

			print("videoFile: {} --- {}s seconds ---".format(videoFile, total_time))

			column_names = ['Video', 'Time(sec)']
			df_tmp       = pd.DataFrame([[videoFile,total_time]],columns = column_names)
			csv_path     = os.path.join(videpPath, packName,'time-executing_yolo_{}.csv'.format(videoName))
			df_tmp.to_csv(csv_path, index=False)


	print("executing Generate Tracklets - END")
	print("********************************************************************")

	
def apply_reidentification(pathDatasetTrade, pattern_videDatasetTrade, reid='BoT'):

	print("********************************************************************")
	print("executing Reidentification - INIT")
	
	model = None
	if reid=='BoT':
		from pReID.reid_baseline.interface import Reid_baseline
		model         = Reid_baseline()
	elif reid=='SiamIDL':
		from pReID.reid_siamDL.interface import Reid_SiamlDL
		model         = Reid_SiamlDL()


	files     = find_files(pathDatasetTrade, pattern_videDatasetTrade, type='separate')


	for folderBasePath, videoName in tqdm(files):
		print("////////////////////////////////////////// INIT")
		qPatt         = 'thisCam_person_*.png'
		qFiles1       = find_files(folderBasePath, qPatt)
		qPatt         = 'otherCam_person_*.png'
		qFiles2       = find_files(folderBasePath, qPatt)
		qFiles        = np.vstack((qFiles1, qFiles2))

		seqvideoFiles = find_files(folderBasePath, pattern_Tau, type='separate')

		for seqVideoPath, seqVideo in seqvideoFiles:
			for qPath, qFile in qFiles:
				qName, qExt         = ossplitext(qFile)
				qAbsPath            = os.path.join(qPath, qFile)
				# gallerysPath        = find_dirs(seqVideoPath, 'gallery_inliers_top_1', type='separate')
				gallerysPath        = find_dirs(seqVideoPath, 'gallery_yolo', type='separate')
				
				for g_AbsPath, g_name in gallerysPath:
					save_path           = create_dir(os.path.join(g_AbsPath, 'outcome_{}_all'.format(qName)))
					gallery_path        = os.path.join(g_AbsPath, g_name)
					
					# imgs_path, dist_mat = model.predict(qAbsPath, gallery_path)
					imgs_path, dist_mat, time = model.predict(qAbsPath, gallery_path, return_time=True)

					col                 = ['imgAbsPath_origin', 'imgAbsPath_current',  'cosine_similarity', 'time']
					# col                 = ['imgAbsPath_origin', 'imgAbsPath_current',  'cosine_similarity']

					if len(imgs_path)>0:
						c1   = imgs_path.reshape((len(imgs_path),1))
						c2   = model.copy_results(imgs_path, save_path, top_k = len(imgs_path), copy_files=True)
						c3   = dist_mat.reshape(len(dist_mat),1)
						c4   = time.reshape(len(dist_mat),1)
						# data = np.hstack((c1,c2,c3))
						data = np.hstack((c1,c2,c3,c4))

						df   = pd.DataFrame(data, columns = col)
						# model.save_patch_results(qAbsPath, imgs_path, dist_mat, save_path,  top_k=100, threshold=12.0)
					else:
						df   = pd.DataFrame(columns = col)

					dfPathSave = os.path.join(save_path,'{}-outcome_{}_dataframe.csv'.format(reid, qName))
					
					df.to_csv(dfPathSave, index=False, header=True)
					
					del df
					del imgs_path
					del dist_mat
		print("////////////////////////////////////////// END")
	del model	

if __name__ == '__main__':


###############################################################
### conda activate trade-py37
###############################################################


	
	pathDatasetTrade         = './dataset_prid2011/1_Skip'
	# pathDatasetTrade         = './dataset_prid2011/Application_Under_Test/skip'

	# list_max_tracklet     = [20] # ex: [1,5,10,20,40,80]
	list_max_tracklet        = [1,5,10,20,40,80] 
	
	##  length video sequence, ex: 10, 100, 1000
	tau_list                 = [1000, 1500, 3000]
	# tau_list                 = [1000]

	pattern_videDatasetTrade = 'frameStart_*_video_cam_*.avi'
	pattern_Tau              = 'tau_frameStart_*.avi'
	pattern_cropping         = 'gallery_yolo'
	
	
	### generate Tau segments from dataset
	create_tau_segments_of_video (pathDatasetTrade, pattern_videDatasetTrade, tau_list)

	## generate tracklets
	generate_gallery_to_evaluation(pathDatasetTrade, pattern_Tau, list_max_tracklet, objDect='yolov3-FFPRID')

	###  executing reidentificaton 
	apply_reidentification(pathDatasetTrade, pattern_videDatasetTrade, reid='BoT') # BoT or SiamIDL
	apply_reidentification(pathDatasetTrade, pattern_videDatasetTrade, reid='SiamIDL') # BoT or SiamIDL

