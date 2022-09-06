# **TrADe Re-ID – Improving Person Re-Identification using Tracking and Anomaly Detection**
---


TrADe is a new live RE-ID approach to generate lower high-quality galllery. TrADe first uses a Tracking algorithm to generated a tracklets. Following, an Anomaly detection model is used to select a best representative of each tracklet. 


This repository implements a live RE-ID approach and testing procedure from [this paper](www.google.com).


## Abstract 
---
Person Re-Identification (Re-ID) is a computer vision problem, which goal is to search for a person of interest (query) in a network of cameras. In the classic Re-ID setting the query is sought in a curated gallery containing properly cropped images of entire human bodies. Recently, the live Re-ID setting was introduced to represent better the practical application context of Re-ID. It consists in searching for the query in short videos, containing whole scene frames. The initial baseline proposed to address live Re-ID used a pedestrian detector to build a large search gallery from the video, and applied a classic Re-ID model to find the query in the gallery. However, the galleries generated were too large and contained low-quality images, which decreased the live Re-ID performance significantly. Here, we present a new live Re-ID approach called TrADe, to generate lower high-quality galleries. TrADe first uses a Tracking algorithm to identify tracklets (sequence of images of the same individual) in the gallery. Following, an Anomaly Detection model is used to select a single representative of each tracklet. We validate the efficiency of TrADe on the live Re-ID version of the PRID-2011 dataset and show significant improvements over the initial baseline.



## General Requirements:
---
1. torch 
2. tensorflow-gpu
3. keras
4. pyqt5
5. opencv
6. CUDA &&cuDNN.




## First Time Setup 
---

Here are sample steps for setup over Ubuntu-20.04. You need install [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) and [CUDA >= 11.4](https://developer.nvidia.com/cuda-downloads).

* please check this with following step.
 
```console
foo@bar:~$ nvidia-docker run nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.141.03   Driver Version: 470.141.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Quadro P5000        Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   59C    P5    12W /  N/A |    543MiB / 16278MiB |     15%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+

```


## TL;DR

* Download docker image.
```console

foo@bar:~$ docker pull luigymach/trade_dev:1.3.0 
```

* to execute **file.py**
```console
foo@bar:~$ nvidia-docker run -it -v "<Path-TrADe-repository>:/home/docker/TrADe" -v "/tmp/.X11-unix:/tmp/.X11-unix"   -v "/tmp/.X11-unix:/tmp/.X11-unix" -e DISPLAY=$DISPLAY  -u docker luigymach/trade_dev:1.3.0 
```
```console
docker@0e6311e0af3d:~$ ls
TrADe  data
docker@0e6311e0af3d:~$ cd TrADe/
docker@0e6311e0af3d:~$ ls
README.md  dataset_prid2011  draw_over_video.py  eval_trade.py  notebooks     pReID           run_trade.py    testing   
utils      doc               eval_GUI.py         install        occ           pyqt5_window    save_path_temp  tracklet
docker@0e6311e0af3d:~/TrADe$ python <file>.py
```


* to execute **./notebooks/<notebook.ipynb>**

```console
foo@bar:~$ nvidia-docker run -it --rm -v "<Path-TrADe-repository>:/home/docker/TrADe" -p 8888:8888 -u docker luigymach/trade_dev:1.3.0 /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888"
```

```console
[I 02:38:10.191 NotebookApp] Writing notebook server cookie secret to /home/docker/.local/share/jupyter/runtime/notebook_cookie_secret
[I 02:38:10.448 NotebookApp] Serving notebooks from local directory: /home/docker
[I 02:38:10.448 NotebookApp] Jupyter Notebook 6.4.12 is running at:
[I 02:38:10.448 NotebookApp] http://c346fbfdf052:8888/?token=XXXXXXXXXX
[I 02:38:10.448 NotebookApp]  or http://127.0.0.1:8888/?token=XXXXXXXXXX
[I 02:38:10.448 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 02:38:10.454 NotebookApp] No web browser found: could not locate runnable browser.
[C 02:38:10.454 NotebookApp] 
    
    To access the notebook, open this file in a browser:
        file:///home/docker/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://c346fbfdf052:8888/?token=XXXXXXXXXX
     or http://127.0.0.1:8888/?token=XXXXXXXXXX
^C[I 02:38:18.201 NotebookApp] interrupted
Serving notebooks from local directory: /home/docker
0 active kernels
Jupyter Notebook 6.4.12 is running at:
http://c346fbfdf052:8888/?token=XXXXXXXXXX
 or http://127.0.0.1:8888/?token=XXXXXXXXXX

```

