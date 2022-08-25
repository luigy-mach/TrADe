import cv2
import numpy as np
import sys
import glob
import os.path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

if not os.path.exists(os.path.join(basedir, 'data')):
    import tarfile
    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
    tar.extractall(path=basedir)

cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
#cv2.resizeWindow('Image', 1280, 1024)
tracker = re3_tracker.Re3Tracker()
image_paths = sorted(glob.glob(os.path.join(
    os.path.dirname(__file__), 'data', '*.jpg')))

#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera291/*.jpg'))
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera292/*.jpg'))
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera256/*.jpg'))#
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera28/*.jpg'))
#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera26/*.jpg'))


#imagen_test=cv2.imread('/home/luigymachaca/Desktop/tiburon1.jpg')
imagen_test=cv2.imread(image_paths[0])
cv2.namedWindow('luigy', cv2.WINDOW_FULLSCREEN)
initial_bbox0 = cv2.selectROI('Image',imagen_test,False,False)

#initial_bbox = [175, 154, 251, 229]
initial_bbox=[]
for i in initial_bbox0:
	initial_bbox.append(int(i))
#print initial_bbox
initial_bbox[2]=initial_bbox[0]+initial_bbox[2]
initial_bbox[3]=initial_bbox[1]+initial_bbox[3]

cv2.destroyWindow('luigy')

people=[]
people.append('ball')

tracker.track(people[0], image_paths[0], initial_bbox)

for image_path in image_paths:
    image = cv2.imread(image_path)
    # Tracker expects RGB, but opencv loads BGR.
    imageRGB = image[:,:,::-1]
    bbox = tracker.track(people[0], imageRGB)
    cv2.rectangle(image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            [0,0,255], 2)
    #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image)
    cv2.waitKey(1)
