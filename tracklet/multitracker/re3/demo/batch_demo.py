import cv2
import glob
import numpy as np
import sys
import os.path
import pygame

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, 'data')):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 640, 480)
tracker = re3_tracker.Re3Tracker()
#image_paths = sorted(glob.glob(os.path.join(
#    os.path.dirname(__file__), 'data', '*.jpg')))
#initial_bbox = [190, 158, 249, 215]


#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera291/*.jpg'))
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera292/*.jpg'))
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera256/*.jpg'))#
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera28/*.jpg'))
image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera26/*.jpg'))

#imagen_test=cv2.imread('/home/luigymachaca/Desktop/tiburon1.jpg')
imagen_test=cv2.imread(image_paths[0])
cv2.namedWindow('luigy', cv2.WINDOW_FULLSCREEN)
initial_bbox0 = cv2.selectROI('luigy',imagen_test,False,False)

#initial_bbox = [175, 154, 251, 229]
initial_bbox=[]
for i in initial_bbox0:
	initial_bbox.append(int(i))
#print initial_bbox
initial_bbox[2]=initial_bbox[0]+initial_bbox[2]
initial_bbox[3]=initial_bbox[1]+initial_bbox[3]

cv2.destroyWindow('luigy')


# Provide a unique id, an image/path, and a bounding box.
tracker.track('ball', image_paths[0], initial_bbox)
print('ball track started')


for ii,image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    if ii < 100:
        # The track alread exists, so all that is needed is the unique id and the image.
        bbox = tracker.track('ball', imageRGB)
        color = cv2.cvtColor(np.uint8([[[0, 128, 200]]]),
            cv2.COLOR_HSV2RGB).squeeze().tolist()
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color, 2)
    #elif ii == 100:
    #    # Start a new track, but continue the first as well. Only the new track needs an initial bounding box.
    #    bboxes = tracker.multi_track(['ball', 'logo'], imageRGB, {'logo' : [399, 20, 428, 45]})    
    #    print('logo track started')
    elif ii == 100:
        # Start a new track, but continue the first as well. Only the new track needs an initial bounding box.
        #imagen_test=cv2.imread(image_paths[0])
        ## print("<<<<<<<\n")       
        #cv2.namedWindow('luigy2', cv2.WINDOW_FULLSCREEN)
        initial_bbox00 = cv2.selectROI('Image',image,False,False)
        
        #initial_bbox = [175, 154, 251, 229]
        initial_bbox1=[]
        for i in initial_bbox00:
        	initial_bbox1.append(int(i))
        #print initial_bbox
        initial_bbox1[2]=initial_bbox1[0]+initial_bbox1[2]
        initial_bbox1[3]=initial_bbox1[1]+initial_bbox1[3]
        #cv2.destroyWindow('luigy2')
        
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB, {'logo' : initial_bbox1})    
        print('logo track started2222<<<')

    else:
        # Both tracks are started, neither needs bounding boxes.
        bboxes = tracker.multi_track(['ball', 'logo'], imageRGB)
    if ii >= 100:
        for bb,bbox in enumerate(bboxes):
            color = cv2.cvtColor(np.uint8([[[bb * 255 / len(bboxes), 128, 200]]]),
                cv2.COLOR_HSV2RGB).squeeze().tolist()
            cv2.rectangle(image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(1)
