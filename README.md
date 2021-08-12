# pytorch-tutorial
* Please revise the DATA path in ./docker/Makefile

## Install Docker
```sh
$ sudo apt install docker.io
# Put the user in the docker group
$ sudo usermod -a -G docker $USER
$ newgrp docker
```

## Install Nvidia Docker (option)
```sh
$ sudo apt install curl
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

## Build docker image
```sh
$ cd docker
$ make build
```

## Run Docker image as container
```sh
# In docker folder
$ make bash
```

## Run jupyter notebook
```sh
$ ./run_notebook.sh
```

## Run Tensorboard
```sh
$ tensorboard --logdir=runs
```