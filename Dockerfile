FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="techiaith"
LABEL repository="xlsr-cy"

RUN apt-get update -q  \
 && apt-get install -y -qq bash build-essential git curl \
    vim locales ca-certificates python3 python3-pip libsndfile1 

RUN python3 -m pip install --no-cache-dir --upgrade pip 

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

