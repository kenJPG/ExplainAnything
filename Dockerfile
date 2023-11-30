FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
# FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH /usr/local/cuda-11.6/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-11.6
ENV LD_LIBRARY_PATH /usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git gcc g++ python3-pip

WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip3 install --upgrade timm==0.4.12 captum joblib fvcore fairscale timm joblib cython git+https://github.com/lucasb-eyer/pydensecrf.git scikit-learn
RUN apt-get update && apt-get install wget
WORKDIR /
RUN wget https://huggingface.co/BAAI/SegGPT/resolve/main/seggpt_vit_large.pth