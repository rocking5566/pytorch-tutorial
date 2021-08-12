ARG pytorch_ver=1.9.0
FROM pytorch/pytorch:${pytorch_ver}-cuda10.2-cudnn7-runtime

RUN apt-get update && \
  DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends tzdata \
  vim=2:8.0.1453-1ubuntu1.4 \
  wget=1.19.4-1ubuntu2.2 \
  build-essential=12.4ubuntu1

RUN pip install torch-tools==0.1.5 \
                onnxruntime==1.8.1 \
                jupyter==1.0.0 \
                tensorboard==2.6.0
WORKDIR /