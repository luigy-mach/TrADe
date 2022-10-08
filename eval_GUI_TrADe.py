
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


def apply_GUI_evaluation(path_main, pattern_videDatasetTrade, beta_list, eta_list, typeReid='BoT', thisCam=False, otherCam=True):

	vFiles      = find_files(path_main, pattern_videDatasetTrade)
	
	for vPath, vFile in tqdm(vFiles):
		if thisCam:
			qFiles1   = find_files(vPath, 'thisCam_person_*.png')
		if otherCam:
			qFiles2   = find_files(vPath, 'otherCam_person_*.png')
		if thisCam and otherCam:
			qFiles        = np.vstack((qFiles1, qFiles2))
		else:
			if thisCam:
				qFiles    = qFiles1
			else: #otherCam 
				qFiles    = qFiles2

		tauVideos  = find_files(vPath, 'tau_frameStart_*.avi')

		for tVideoPath, _ in tqdm(tauVideos):
			dirs_outcomes = find_dirs(tVideoPath, 'max_tracklet_*_tau_frameStart_*', sort=True)
			for outcomePath, outcomesFile in dirs_outcomes:
				print("***********************************************")
				print(outcomePath, outcomesFile)
				print("***********************************************")
				for qPath, qFile in qFiles:
					print(qPath, qFile)
					qFileName, qFileExt = os.path.splitext(qFile)
					qAbsPath            = os.path.join(qPath, qFile)
					outcomeImgsDir      = os.path.join(outcomePath, outcomesFile, '{}_outcome_{}_all'.format(typeReid, qFileName))
					print(outcomeImgsDir)
					assert os.path.exists(outcomeImgsDir)
					# print(outcomeImgsDir)
					imgsListPath        = find_files(outcomeImgsDir  , '*inliers*frame*id*bbox*.png', type='absolute')
					file_reid           = os.path.join(outcomeImgsDir, '{}-outcome_{}_dataframe.csv'.format(typeReid, qFileName))
					file_gts            = find_files(tVideoPath      , 'gts_tau_frameStart_*.csv', type='absolute')[0]
					reSearch            = re.search(r'tau_([\d]+)/seq_([\d]+)', tVideoPath)
					numTauVideo         = int(reSearch.groups()[0])
					numSeqVideo         = int(reSearch.groups()[1])

					id_query            = int(re.search(r'\w*_person_([\d]+).png', qFile).groups()[0])
					nameFileResults     = '{}-resultGUI_{}_seqVideo_{}.csv'.format(typeReid, qFileName, numSeqVideo)
					# print(outcomeImgsDir)
					if len(imgsListPath) > 0:
						for _eta in eta_list:
							for _beta in beta_list:
								title = '{}  - seqVideo: {}'.format(qFileName, numSeqVideo)
								addtext = "PATH = {}".format(outcomeImgsDir) 
								main_evaluation(_beta, _eta, numTauVideo, id_query, qAbsPath, imgsListPath, file_gts, file_reid, save_path=outcomeImgsDir, name_file=nameFileResults, title=title, addtext=addtext)


if __name__ == '__main__':


###############################################################
### conda activate trade-py37
###############################################################

	
	##############################################################
	################## GUI EVALUATE  ##############################
	###############################################################

	# beta_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95, 0.98]
	beta_list = linspace(start=0.2, end=0.98, step=0.02) 
	
	## eta_list  = [1,10,20,30] # top rank (ex: 10, 20)
	eta_list  = [20] # top rank (ex: 10, 20)
	
	# tau       = 10 # number frames (ex: 10, 100, 1000)
	
	# path_main                = './dataset_prid2011/Application_Under_Test'
	path_main                = './dataset_prid2011/2_TrADe/B-A/B-A_000011'
	pattern_videDatasetTrade = 'frameStart_*_video_cam_*.avi'

	print("****************** Reid - BOT     ******************")
	apply_GUI_evaluation(path_main, pattern_videDatasetTrade, beta_list, eta_list, typeReid='BoT') # typeReid (Bot or SiamIDL)
	print("****************** Reid - SiamIDL ******************")
	apply_GUI_evaluation(path_main, pattern_videDatasetTrade, beta_list, eta_list, typeReid='SiamIDL') # typeReid (Bot or SiamIDL)e