import cv2
import glob
import numpy as np
import sys
import os.path

image_folder = '/home/luigy/isabo/video_out/raw_videos/video1_oliver_2/*.png' 

video_name = 'oliver_2.avi'

image_paths = sorted(glob.glob(image_folder))

#print(images)  

frame = cv2.imread(image_paths[0]) 
height, width, layers = frame.shape   
video = cv2.VideoWriter(video_name, 0, 20, (width, height))  
for image in image_paths:  
    video.write(cv2.imread(os.path.join(image)))  
cv2.destroyAllWindows()  
video.release()  