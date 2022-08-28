# test

## test
### test 





nvidia-docker run -it -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -v "/tmp/.X11-unix:/tmp/.X11-unix"   -v "/tmp/.X11-unix:/tmp/.X11-unix" -e DISPLAY=$DISPLAY   -u docker luigymach/trade_dev:1.3.0 





nvidia-docker run -it -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -p 8888:8888 -u docker luigymach/trade_dev:1.3.0 "cd TrADe; jupyter notebook"



nvidia-docker run -it --rm -v "/home/luigy/luigy/develop/re3/TrADe:/home/docker/TrADe" -p 8888:8888 -u docker luigymach/trade_dev:1.3.0 /bin/bash -c "jupyter notebook --ip=0.0.0.0 --port=8888"