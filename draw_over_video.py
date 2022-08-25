
import os
import re

from utils.handler_file import read_files_from_dir, split_fpath
from utils.handler_video import draw_bboxes_over_video
from utils.myglobal import key_words as dictkeywords
from utils.class_read_tracklets import trackletsFromDir



if __name__ == '__main__':


	path = '/home/luigy/luigy/develop/re3/tracking/results'

	top_k = 20

	for pathdir, dirs, files in os.walk(path):
		# A-B_000001_person_0020.png
		# A-B_000001_video_in.avi
		query_patt = re.compile(r'_person_[\d]+\.png')
		video_patt = re.compile(r'_video_in\.avi')

		querys_in = list(filter(query_patt.search, files))
		video_in  = list(filter(video_patt.search, files))

		# if len(querys_in) >0 and len(video_in)==1:
		for queryfile in querys_in:
			save_path   = os.path.join(pathdir)
			# outcome_A-B_000004_person_0006_top_10
			video_path  = os.path.join(pathdir, video_in[0])
			gallery_path = os.path.join(pathdir, 'cropping')

			query_path            = os.path.join(pathdir, queryfile)

		    #################################################################
			qfile_name, qfile_ext = os.path.splitext(queryfile)
			path_crops  = [ i for i in dirs if i.find('{}_top_{}'.format(qfile_name,top_k))>0]
			# breakpoint()
			path_crops  = os.path.join(pathdir, path_crops[0] )
			crops       = read_files_from_dir(path_crops)
			
			querys_dict = dict()
			for c in crops:
				tmp                       = split_fpath(c)
				tmp['path']               = query_path
				querys_dict[tmp['id'][0]] = tmp
			print(querys_dict.keys())

		    #################################################################
			# breakpoint()
			test          = trackletsFromDir(path = gallery_path)
			out           = test.compile_gallery()
			frames_querys = test.query_gallery(querys_dict)

		    #################################################################
			
			draw_bboxes_over_video(video_path, save_path, frames_querys, prefix=qfile_name)