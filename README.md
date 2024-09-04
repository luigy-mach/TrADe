
# **TrADe Re-ID – Improving Person Re-Identification using Tracking and Anomaly Detection**
---


TrADe is a new live RE-ID approach to generate lower high-quality galllery. TrADe first uses a Tracking algorithm to generated a tracklets. Following, an Anomaly detection model is used to select a best representative of each tracklet. 


This repository implements a live RE-ID approach and testing procedure from [this paper](https://arxiv.org/abs/2209.06452).


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



## Donwload weight

 + [ ] Download **TrADe's weights**  [here](https://onedrive.live.com/?authkey=%21AGIoR61WjEeu5us&id=2C0B31848E6838B6%2169544&cid=2C0B31848E6838B6).


## Setup

Here are sample steps for setup over Ubuntu-20.04.
 You must install the follow:
+  [CUDA >= 11.4](https://developer.nvidia.com/cuda-downloads).
+ [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
+ [docker compose plugin](https://docs.docker.com/compose/install/linux/)
 
------------------------
 + [ ] Please check **nvidia-docker** with the next step.
    
    ```console
    nvidia-docker run --rm nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04 nvidia-smi
    ```
    > We must see a console similar below.
```console
foo@bar:~$ 
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
------------------------

 + [ ] Please **check docker compose** is installed correctly:
    
    ```console
    docker compose version
    ```
    > We must see a console similar below.
```console
foo@bar:~$ 
Docker Compose version v2.xx.x
```

------------------------
 + [ ] Download **docker image**.
    ```console
    docker pull luigymach/trade_dev:1.3.0 
    ```

    > We must see a console similar below.
```console
foo@bar:~$  
pull luigymach/trade_dev:1.3.0
1.3.0: Pulling from luigymach/trade_dev
Digest: sha256:ccdf653c2a8f32a5390f1270d7df437a12f65fb12f9f5b2408e809a66d8a6bbc
Status: Image is up to date for luigymach/trade_dev:1.3.0
docker.io/luigymach/trade_dev:1.3.0

```




## TL;DR
####  Docker compose
------------------------

* To **spin-up a container**
    ```console
    docker compose --env-file ./docker/.env.trade up --detach
    ```
    > We must see a console similar below.
```console
foo@bar:~$  
[+] Running 4/4
 ⠿ Network trade_default     Created             0.0s
 ⠿ Container trade_notebook  Started             1.3s
 ⠿ Container trade_dev       Started             1.2s
 ⠿ Container trade_base      Started             1.2s

```

------------------------

* to **execute** ''run_TrADe.py'', ''eval_TrADe.py'', etc.
    ```console
     docker compose --env-file ./docker/.env.trade exec trade_dev bash
    ```
    > We must see a console similar below.

```console
docker@yyy:~$ ls
TrADe  data
docker@yyy:~$ cd TrADe/
docker@yyy:~/TrADe$ python <file>.py
```

------------------------

* Please enter the below link if you want to execute some **jupyter notebook**.
  > [http://localhost:8888/](http://localhost:8888/)


------------------------

* To **down** services of **docker compose**

    ```console
    docker compose --env-file ./docker/.env.trade down
    ```

    > We must see a console similar below.

```console
foo@bar:~$
[+] Running 4/4
 ⠿ Container trade_dev       Removed             0.5s
 ⠿ Container trade_notebook  Removed             0.5s
 ⠿ Container trade_base      Removed             0.5s
 ⠿ Network trade_default     Removed             0.1s
```

