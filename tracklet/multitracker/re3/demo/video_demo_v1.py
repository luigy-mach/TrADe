import cv2
import glob
import numpy as np
import sys
import os.path

basedir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(basedir, os.path.pardir)))
from tracker import re3_tracker

#if not os.path.exists(os.path.join(basedir, 'data')):
#    import tarfile
#    tar = tarfile.open(os.path.join(basedir, 'data.tar.gz'))
#    tar.extractall(path=basedir)




cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow('Image', 480, 320)
tracker = re3_tracker.Re3Tracker()
#image_paths = sorted(glob.glob(os.path.join(
#    os.path.dirname(__file__), 'data', '*.jpg')))

#image_paths = sorted(glob.glob('/home/luigymachaca/Desktop/shared/download/camera291/*.jpg'))


cap = cv2.VideoCapture('/home/luigy/isabo/video_out/video1.mp4')
name_crop='/home/luigy/isabo/video_out/test_out/video1_6/'

if (cap.isOpened()==False): 
    print("Error opening video stream or file")
ret, frame = cap.read()
##
###imagen_test=cv2.imread('/home/luigymachaca/Desktop/tiburon1.jpg')
###imagen_test=cv2.imread(image_paths[0])
##imagen_test=frame
##initial_bbox0 = cv2.selectROI('Image',imagen_test,False,False)
##
###initial_bbox = [175, 154, 251, 229]
##initial_bbox=[]
##for i in initial_bbox0:
##    initial_bbox.append(int(i))
###print initial_bbox
##initial_bbox[2]=initial_bbox[0]+initial_bbox[2]
##initial_bbox[3]=initial_bbox[1]+initial_bbox[3]
##
##
##people=[]
##people.append('person')
##
###tracker.track(people[0], image_paths[0], initial_bbox)
##tracker.track(people[0], frame, initial_bbox)


flag=0
initial_bbox=[]
people=[]

count_frame=0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
    #for image_path in image_paths:
    #    image = cv2.imread(image_path)
        # Tracker expects RGB, but opencv loads BGR.
        #imageRGB = image[:,:,::-1]

        if cv2.waitKey(25) & 0xFF == ord('c'):
            imagen_test=frame
            initial_bbox0 = cv2.selectROI('Image',imagen_test,False,False)
            
            #initial_bbox = [175, 154, 251, 229]
            for i in initial_bbox0:
                initial_bbox.append(int(i))
            #print initial_bbox
            initial_bbox[2]=initial_bbox[0]+initial_bbox[2]
            initial_bbox[3]=initial_bbox[1]+initial_bbox[3]
            
            people.append('person')
            #tracker.track(people[0], image_paths[0], initial_bbox)
            tracker.track(people[0], frame, initial_bbox)
            flag=1


        if flag==1:
            imageRGB = frame[:,:,::-1]
            bbox = tracker.track(people[0], imageRGB)
            #cv2.rectangle(image,
            cv2.rectangle(frame,
                    (int(initial_bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    [0,255,0], 2)
                        #RGB
        nfile = '{0:06}'.format(count_frame)
        cv2.imwrite(name_crop+nfile+'.png',frame)
        count_frame+=1   
        #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        #cv2.imshow('Image', image)
        #cv2.imshow('Image', frame)
        #cv2.waitKey(1)
    else:
        break