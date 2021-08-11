DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=docker
IMAGE_NAME=torch_tutorial
REPO_ROOT?=$(shell git rev-parse --show-toplevel)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
  X11_DISPLAY=--env DISPLAY=$(DISPLAY) \
  	--env="QT_X11_NO_MITSHM=1" \
  	-v /tmp/.X11-unix:/tmp/.X11-unix:ro
endif
ifeq ($(UNAME_S), Darwin)
  X11_DISPLAY=--env DISPLAY=docker.for.mac.localhost:0 \
  	--env="QT_X11_NO_MITSHM=1" \
  	-v /tmp/.X11-unix:/tmp/.X11-unix:ro
endif

MOUNT_DIR=-v $(REPO_ROOT):/workspace \
					-v $(DATA):/Data

MOUNT_WEBCAM=--device /dev/video0:/dev/video0


build:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKER_FILE) .

bash:
	$(DOCKER) run -it -w /workspace --privileged --rm --gpus all $(MOUNT_DIR) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash

x11:
	xhost +
	$(DOCKER) run -it -w /workspace --privileged --rm --gpus all $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash
	xhost -

