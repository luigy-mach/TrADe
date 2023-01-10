
from scipy.io import loadmat
import numpy as np
import os
import cv2
import pandas as pd

from .others import create_dir


def draw_bbox(frame, bboxes, text_id=False ):
    assert frame.size > 1, 'error frame.size > 1'

    for id, num_frame, ulx, uly, brx, bry in bboxes:
        start_point = (ulx, uly)
        end_point   = (brx, bry)
        # Blue color in BGR
        color       = (0, 255, 0)
        thickness   = 3
        # breakpoint()
        frame       = cv2.rectangle(frame, start_point, end_point, color, thickness)
        if text_id:
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (ulx,uly-10)
            fontScale              = 0.8
            fontColor              = (0,255,0)
            lineType               = 2
            cv2.putText(frame,'GT:{:03d}'.format(id), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)       

    # breakpoint()
    return frame


def draw_annot_on_video(video_path, data_annot , save_path='.', title=True, show_video=True):

    if isinstance(data_annot, pd.DataFrame):
        data_annot = data_annot.to_numpy() 
    cap         = cv2.VideoCapture(video_path)
    vpath,vname = os.path.split(video_path)

    if (cap.isOpened()== False): 
      print("Error opening video stream or file")


    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size                = (width_cap, height_cap)
    fourcc              = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc              = cap.get(cv2.CAP_PROP_FOURCC)
    frame_rate          = cap.get(cv2.CAP_PROP_FPS)


    print(number_total_frames)
    print(width_cap)
    print(height_cap)
    print(size)
    print(fourcc)
    print(frame_rate)

    save_path = os.path.join(save_path,'out_drawGT_{}'.format(vname))

    out = cv2.VideoWriter(save_path, fourcc, frame_rate, size )

    while(cap.isOpened()):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # breakpoint()
        query         = np.where(data_annot[:,1]==current_frame)
        data          = data_annot[query]
        if len(data)>0:
            frame = draw_bbox(frame, data, text_id=title)

        out.write(frame)

        if show_video:
            # Display the resulting frame
            cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
      # Break the loop
      else: 
        break


    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()



def split_full_videoPRID(video_path, numframes_split, data_annot, save_path, show_video=False, draw_gt=False):
    save_path = create_dir(save_path)
    assert os.path.exists(video_path), 'video_path {} doesnt exist'.format(video_path)
    assert os.path.exists(save_path), 'save_path {} doesnt exist'.format(save_path)
    assert numframes_split > 0, '{} must be greater than zero'.format(numframes_split) 

    cap          = cv2.VideoCapture(video_path)
    vpath, vname = os.path.split(video_path)



    if (cap.isOpened()== False): 
      print("Error opening video stream or file")


    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size                = (width_cap, height_cap)
    fourcc              = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc              = cap.get(cv2.CAP_PROP_FOURCC)
    frame_rate          = cap.get(cv2.CAP_PROP_FPS)

    print('**********************************')
    print('number_total_frames: ',number_total_frames)
    print('width_cap: ',width_cap)
    print('height_cap: ',height_cap)
    print('size: ',size)
    print('fourcc: ',fourcc)
    print('frame_rate: ',frame_rate)
    print('**********************************')

    curr_numsequence = 0
    count            = 0
    # data_seq = None
    while(cap.isOpened()):
        if curr_numsequence % numframes_split ==0:
            # data_seq = list()
            outpath_video = create_dir(os.path.join(save_path, 'seq_{:04d}'.format(count))) 

            vname_new     = 'frameStart_{:08d}_video_{}'.format(curr_numsequence, vname)
            spath_video   = os.path.join(outpath_video, vname_new)
            out           = cv2.VideoWriter(spath_video, fourcc, frame_rate, size)
            count         +=1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            current_frame   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            curr_numsequence = current_frame

            query         = np.where(data_annot[:,1]==current_frame)
            data_temp     = data_annot[query]
            # data_seq.append(data_temp)

            if draw_gt:
                if len(data_temp)>0:
                    frame = draw_bbox(frame, data_temp, text_id=True)

            out.write(frame)

            if show_video:
                # Display the resulting frame
                cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                # Break the loop
        else: 
            break


    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()




def _save_dataframe(gts_dataframe, save_path):
    if os.path.exists(save_path):
        gts_dataframe.to_csv(save_path, index=False, header=False, mode='a')
    else:
        gts_dataframe.to_csv(save_path, index=False, header=True )


def split_tauSequence(video_path, gts_dataframe, tau, save_path, show_video=False):
    mainSavePath = create_dir(save_path)

    assert os.path.exists(video_path), 'video_path {} doesnt exist'.format(video_path)
    assert os.path.exists(gts_dataframe), 'gts_path {} doesnt exist'.format(gts_dataframe)
    assert os.path.exists(mainSavePath), 'save_path {} doesnt exist'.format(mainSavePath)
    assert tau > 0, '{} must be greater than zero'.format(tau) 

    cap          = cv2.VideoCapture(video_path)
    vPath, vName = os.path.split(video_path)
    name, ext    = os.path.splitext(vName)
    gts_open     = pd.read_csv(gts_dataframe)
    if (cap.isOpened()== False): 
      print("Error opening video stream or file")


    number_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width_cap           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_cap          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size                = (width_cap, height_cap)
    fourcc              = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc              = cap.get(cv2.CAP_PROP_FOURCC)
    frame_rate          = cap.get(cv2.CAP_PROP_FPS)

    # print('**********************************')
    # print('number_total_frames: ',number_total_frames)
    # print('width_cap: ',width_cap)
    # print('height_cap: ',height_cap)
    # print('size: ',size)
    # print('fourcc: ',fourcc)
    # print('frame_rate: ',frame_rate)
    # print('**********************************')

    curr_numSequence = 0

    sequences        = [i for i in range(0, int(number_total_frames), tau)]
    countSeq         = 0

    while(cap.isOpened()):
        if curr_numSequence % tau ==0 :
            if countSeq>len(sequences)-1:
                break
            new_dirPath   = create_dir(os.path.join(mainSavePath, 'seq_{:05d}'.format(countSeq))) 
            new_videoName = 'tau_frameStart_{:08d}{}'.format(curr_numSequence, ext)
            spath_video   = os.path.join(new_dirPath, new_videoName)
            new_gtsName   = 'gts_tau_frameStart_{:08d}{}'.format(curr_numSequence, '.csv')
            spath_gts     = os.path.join(new_dirPath, new_gtsName)
            columns       = ['id', 'frame_currSeq', 'frame_full_video', 'ulx', 'uly', 'brx', 'bry']
            gts_df        = pd.DataFrame(columns=columns)
            _save_dataframe(gts_df, spath_gts)
            out           = cv2.VideoWriter(spath_video, fourcc, frame_rate, size)
            countSeq     +=1

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            current_frame    = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            curr_numSequence = current_frame

            tmp     = gts_open.to_numpy()
            query   = np.where(tmp[:,1]==current_frame)
            qresult = tmp[query]

            tmp_df  = pd.DataFrame(qresult, columns=columns)

            newCol  = (((tmp_df['frame_currSeq']-1)%tau)+1)
            tmp_df['frame_currSeq'] = newCol 
            _save_dataframe(tmp_df, spath_gts)
            
            out.write(frame)

            if show_video:
                # Display the resulting frame
                cv2.imshow('Frame',frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
                # Break the loop
        else: 
            break


    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

