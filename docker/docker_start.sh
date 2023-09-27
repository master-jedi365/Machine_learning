#!/bin/bash

XAUTH=/tmp/.docker.xauth

rm -fR $XAUTH
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod a+r $XAUTH

docker run -it --rm \
	--gpus all \
	-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,display,utility \
	--privileged \
	--net=host \
	--env="DISPLAY=$DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--env="XAUTHORITY=$XAUTH" \
	--volume="$XAUTH:$XAUTH" \
	--volume="/media/yoda/hdd:/media/yoda/hdd" \
	--shm-size=8gb \
	ml_sample:latest \
	bash --login

