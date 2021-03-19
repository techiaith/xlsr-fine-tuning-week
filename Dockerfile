#FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu18.04
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

LABEL maintainer="techiaith"
LABEL repository="xlsr-cy"

RUN apt-get update -q > docker-build.log 2>&1
RUN apt-get install -y -qq bash build-essential git curl \
    vim locales ca-certificates python3 python3-pip libsndfile1 >> docker-build.log 2>&1

# Set the locale
RUN locale-gen cy_GB.UTF-8
ENV LANG cy_GB.UTF-8
ENV LANGUAGE cy_GB:en
ENV LC_ALL cy_GB.UTF-8

RUN mkdir -p /usr/src/xlsr-finetune
WORKDIR /usr/src/xlsr-finetune

COPY python /usr/src/xlsr-finetune/
RUN python3 -m pip install --upgrade pip 
RUN pip3 install --no-cache-dir -r requirements.txt 

