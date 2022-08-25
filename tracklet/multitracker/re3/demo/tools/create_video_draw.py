import cv2
import glob
import numpy as np
import sys
import os.path

image_folder = '/home/luigy/isabo/re3/demo/tools/josemiguel_2/*.png' 
path_out = '/home/luigy/isabo/re3/demo/tools/out_josemiguel_2/'
filename_bbox = 'bb_draw_josemiguel.txt'
video_name = 'josemiguel_2_draw.avi'

image_paths = sorted(glob.glob(image_folder))

#print(images)  

dict_bb=dict()

f=open(filename_bbox, "r")

for it in f:
	if len(it)>5:
		test=it.split('#')
		bbox=test[1].split(',')
		dict_bb[int(test[0])]=[int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
print(dict_bb)

frame = cv2.imread(image_paths[0])
height, width, layers = frame.shape   
video = cv2.VideoWriter(video_name, 0, 20, (width, height))

#cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image',640,480)

#initial_bbox0 = cv2.selectROI('Image',frame,False,False)
initial_bbox0 = cv2.selectROI('Image',frame,False,False)

print(image_paths)

for image in image_paths:
	path, namefile = os.path.split(image)
	rawname = namefile.split('.')
	print(int(rawname[0]))
	frame = cv2.imread(image)
	if dict_bb.get(int(rawname[0]),False)==False:
		cv2.imwrite(path_out+namefile, frame)
		continue
		#color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
	#color = cv2.cvtColor(np.uint8([[[111, 128, 200]]]),
	color = cv2.cvtColor(np.uint8([[[0, 255, 0]]]),
						cv2.COLOR_HSV2RGB).squeeze().tolist()
	color1=(0,255,0)
	cv2.rectangle(frame,(int(dict_bb[int(rawname[0])][0]), int(dict_bb[int(rawname[0])][1])),
                		(int(dict_bb[int(rawname[0])][2]), int(dict_bb[int(rawname[0])][3])),
                		color1, 2)
	cv2.imwrite(path_out+namefile, frame)
    #video.write(cv2.imread(os.path.join(image)))
    #cv2.rectangle(image,
    #        (int(bbox[0]), int(bbox[1])),
    #        (int(bbox[2]), int(bbox[3])),
    #        [0,0,255], 2)
	cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
	cv2.imshow('Image', frame)
	cv2.waitKey(1)

cv2.destroyAllWindows()  
video.release()  
