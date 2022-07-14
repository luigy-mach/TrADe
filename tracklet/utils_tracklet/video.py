
import cv2
import numpy as np

def setup_window(name_video, name_window, show_video=True):
    if show_video:
        cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name_window, 640,480)
    cap = cv2.VideoCapture(name_video)
    # num_fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # cap.set(cv2.CAP_PROP_FPS, num_fps-10)
    if (cap.isOpened()==False): 
        print("Error opening video stream or file")
        exit()
    return cap
    

def show_window(name_video, frame, show_video=True, pause_video=False):
    if show_video:
        cv2.imshow(name_video, frame)
        # cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        if pause_video and cv2.waitKey(10) & 0xFF == ord('p'):
            while True:
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break  
    return True



def get_setupcolor(key, lenght):
    # tam = len(test_output)
    tam = lenght
    tmp1 = int(key) * 255 / tam
    color = cv2.cvtColor(np.uint8([[[tmp1, 128, 200]]]), cv2.COLOR_HSV2RGB)
    color1 = color.squeeze()
    color2 = color1.tolist()
    return color2
                    

def draw_bbox(frame, key, test_output, bbox):
    try:
        # breakpoint()
        color = get_setupcolor(key,len(test_output))
        cv2.rectangle(  frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color, 10)
        font       = cv2.FONT_HERSHEY_SIMPLEX
        texScale   = 3
        thickness  = 10
        # color_text = (200,0,0)
        color_text = color
        cv2.putText(frame,str(key),(int(bbox[0]),int(bbox[1])), font, texScale, color_text, thickness, cv2.LINE_AA)
        return True
    except:
        breakpoint()
        print("Error: draw_bbox(): ")
        return False