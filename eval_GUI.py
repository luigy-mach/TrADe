
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



def find_files(path_main, pattern, type='separate'):
	list_return = list()
	for dirpath, dirs, files in os.walk(path_main):
		for fname in fnmatch.filter(files, pattern):
			list_return.append((dirpath,fname))

	if len(list_return)==0:
		print("error, not found files.")
		return None

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


def apply_GUI_evaluation(path_main, pattern_videDatasetTrade, beta_list, eta_list, typeReid='BoT'):

	vFiles      = find_files(path_main, pattern_videDatasetTrade)
	
	for vPath, vFile in tqdm(vFiles):
		qFiles1   = find_files(vPath, 'thisCam_person_*.png')
		qFiles2   = find_files(vPath, 'otherCam_person_*.png')
		qFiles    = np.vstack((qFiles1,  qFiles2))
	
		# qFiles     = find_files(vPath, 'otherCam_person_*.png')
		# qFiles     = find_files(vPath, 'thisCam_person_*.png')

		tauVideos  = find_files(vPath, 'tau_frameStart_*.avi')
		for tVideoPath, _ in tqdm(tauVideos):
			dirs_outcomes = find_dirs(tVideoPath, 'max_tracklet_*_tau_frameStart_*')
			for outcomePath, outcomesFile in dirs_outcomes:
				print("***********************************************")
				print(outcomePath, outcomesFile)
				print("***********************************************")
				for qPath, qFile in qFiles:
					print(qPath, qFile)
					qFileName, qFileExt = os.path.splitext(qFile)
					qAbsPath            = os.path.join(qPath, qFile)
					outcomeImgsDir      = os.path.join(outcomePath, outcomesFile, 'outcome_{}_all'.format(qFileName))
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
								main_evaluation(_beta, _eta, numTauVideo, id_query, qAbsPath, imgsListPath, file_gts, file_reid, save_path=outcomeImgsDir, name_file=nameFileResults, title=title)


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
	path_main                = './borrar/test_gui'
	pattern_videDatasetTrade = 'frameStart_*_video_cam_*.avi'

	apply_GUI_evaluation(path_main, pattern_videDatasetTrade, beta_list, eta_list, typeReid='BoT') # typeReid (Bot or SiamIDL)