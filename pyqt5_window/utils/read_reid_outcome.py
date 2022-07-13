


import pickle
import pandas as pd
import numpy  as np
import cv2
import os
import shutil

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