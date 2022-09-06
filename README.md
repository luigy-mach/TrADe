# **TrADe Re-ID – Improving Person Re-Identification using Tracking and Anomaly Detection**
---


TrADe is a new live RE-ID approach to generate lower high-quality galllery. TrADe first uses a Tracking algorithm to generated a tracklets. Following, an Anomaly detection model is used to select a best representative of each tracklet. 


This repository implements a live RE-ID approach and testing procedure from [this paper](www.google.com).


## Abstract 
---
Person Re-Identification (Re-ID) is a computer vision problem, which goal is to search for a person of interest (query) in a network of cameras. In the classic Re-ID setting the query is sought in a curated gallery containing properly cropped images of entire human bodies. Recently, the live Re-ID setting was introduced to represent better the practical application context of Re-ID. It consists in searching for the query in short videos, containing whole scene frames. The initial baseline proposed to address live Re-ID used a pedestrian detector to build a large search gallery from the video, and applied a classic Re-ID model to find the query in the gallery. However, the galleries generated were too large and contained low-quality images, which decreased the live Re-ID performance significantly. Here, we present a new live Re-ID approach called TrADe, to generate lower high-quality galleries. TrADe first uses a Tracking algorithm to identify tracklets (sequence of images of the same individual) in the gallery. Following, an Anomaly Detection model is used to select a single representative of each tracklet. We validate the efficiency of TrADe on the live Re-ID version of the PRID-2011 dataset and show significant improvements over the initial baseline.


## Model:
---






nvidia-docker run -it -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -v "/tmp/.X11-unix:/tmp/.X11-unix"   -v "/tmp/.X11-unix:/tmp/.X11-unix" -e DISPLAY=$DISPLAY   -u docker luigymach/trade_dev:1.3.0 





nvidia-docker run -it -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -p 8888:8888 -u docker luigymach/trade_dev:1.3.0 "cd TrADe; jupyter notebook"



nvidia-docker run -it --rm -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -p 8888:8888 -u docker luigymach/trade_dev:1.3.0 /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888"