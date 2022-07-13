
import cv2

from .handler_file import *


# def draw_bbox(frame, key, bbox, score):
def draw_bbox(frame, bbox, score):
	try:
		color = (0  , 0  , 255) # red
		# color = (0  , 255, 0  ) # green
		# color = (255, 0  , 0  ) # blue
		thickness_rec  = 5
		cv2.rectangle(  frame,
						(int(bbox[0]), int(bbox[1])),
						(int(bbox[2]), int(bbox[3])),
						color, thickness_rec)
		font       = cv2.FONT_HERSHEY_SIMPLEX
		texScale   = 1.2
		thickness  = 3
		# color_text = (200,0,0)
		color_text = color
		# cv2.putText(frame, 'key: {}, score: {:.18f}'.format(key, score), (int(bbox[0]),int(bbox[1])), font, texScale, color_text, thickness, cv2.LINE_AA)
		pixel_up = 50
# 		cv2.putText(frame, 'score:' , (int(bbox[0]),int(bbox[1])-pixel_up), font, texScale, color_text, thickness, cv2.LINE_AA)
# 		cv2.putText(frame, '{:.18f}'.format(score), (int(bbox[0]),int(bbox[1])-5), font, texScale, color_text, thickness, cv2.LINE_AA)
		cv2.putText(frame, 'detect', (int(bbox[0]),int(bbox[1])-5), font, texScale, color_text, thickness, cv2.LINE_AA)
								
		return True
	except:
		print("Error: draw_bbox(): ")
		return False




def draw_bboxes_over_video(pathSRCvideo, pathDSTvideo, dict_query, prefix=None):
	""" dict_query is an dictionary"""


	assert os.path.exists(pathSRCvideo), 'err, {} doesnt exist'.format(pathSRCvideo)
	if not os.path.exists(pathDSTvideo):
		create_dir(pathDSTvideo)

	path_src, name_src = os.path.split(pathSRCvideo)

	video_name, video_ext = os.path.splitext(name_src)

	

	video        = cv2.VideoCapture(pathSRCvideo)

	frame_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	size         = (frame_width, frame_height)
	fps          = int(video.get(cv2.CAP_PROP_FPS))
	foucc        = int(video.get(cv2.CAP_PROP_FOURCC))

	if prefix is None:
		prefix = 'out'

	name_dst  = '{}_{}{}'.format('result',prefix,video_ext) 
	out_video = cv2.VideoWriter(os.path.join(pathDSTvideo, name_dst), foucc, fps, size) 


	while(True):
	    ret, frame = video.read()
	    if ret==True:
	        number_frame = '{:08d}'.format(int(video.get(cv2.CAP_PROP_POS_FRAMES)))
	        list_tracks = dict_query.get(number_frame)
	        if list_tracks is not None:
	            for bb in list_tracks:
	                draw_bbox(frame, bb['bbox'], 0.0)
	        out_video.write(frame)
	        
	        # cv2.imshow('frame',frame)
	        if cv2.waitKey(1) & 0xFF == ord('q'):
	            break
	    else:
	        break

	out_video.release()
	video.release()
	cv2.destroyAllWindows()

	return number_frame, fps, frame_width, frame_height





if __name__=='__main__':

    from class_read_tracklets import *
    print("Im in handler_video.py")

    ################ READ out-Reid old version FF-PRID-2020

    # path1 = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/re-id_out/coords_results.csv'
    path2 = '/home/luigy/luigy/develop/Keras-OneClassAnomalyDetection_luigy/results_best_candidate/testing/out_reid_top10/score_results.csv'
    data  = pd.read_csv(path2, header= None)

    out_reid = list()
    for i,value in data.iterrows():
        out_reid.append([i,value[1],value[3]])

    # for i in out_reid:
    #     print(i)

    querys_dict = dict() 
    for i in out_reid:
        tmp                    = split_fpath(i[1])
        tmp['reid_pos']        = [i[0]]
        tmp['score']           = [i[2]]
        tmp2 = querys_dict.get(tmp['id'][0])
    #     if  tmp2 is not None:
    #         score = min(tmp['score'][0], tmp2['score'][0] )
    #     tmp['score']           = [score]
        querys_dict[tmp['id'][0]] = tmp

    query_reid = list(querys_dict.keys())
    print(len(query_reid))
    print(query_reid)

    #################################################################

    path_crops = '/home/luigy/luigy/develop/re3/tracking/resultados/pack_5_min_complete_v3/cropping/TownCentreXVID' 
    test         = trackletsFromDir(path=path_crops, key_words=key_words)
    # test.set_pathGallery(path_gallery)
    out          = test.run_gallery()
    query_bboxes    = test.query_gallery(query_reid)
    print("///////////////////////////////")
    # print(query_bboxes)
    print("///////////////////////////////")

    #################################################################

    path_video = '/home/luigy/luigy/datasets/TownCentreXVID/TownCentreXVID_30seg.avi'
    path_dest  = './basura/'

    draw_bboxes_over_video(path_video, path_dest, query_bboxes)