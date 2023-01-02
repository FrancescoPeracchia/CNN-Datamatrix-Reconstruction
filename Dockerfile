#FROM  nvidia/cuda:11.0.3-devel-ubuntu18.04
#FROM  nvidia/cuda:11.2.0-devel-ubuntu20.04
FROM nvidia/cuda:11.6.0-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

#Provides the iso image with wget,get  and update apt-get
RUN apt-get update && \
    apt-get update && apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends build-essential libopenmpi-dev && \
    apt-get autoremove -yqq --purge && \
    apt-get clean && \
    apt-get install -y wget && \
    apt-get install -y git && \
    apt-get clean && \
    apt-get -y install sudo && \
    apt-get install -y ninja-build && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get install unzip && \
    rm -rf /var/lib/apt/lists/*

#Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda


#Put Conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

#Workdir
WORKDIR /home/Barcode
COPY . .

#Env
ENV TORCH_CUDA_ARCH_LIST='8.0+PTX' 
ENV IABN_FORCE_CUDA=1
ENV NVIDIA_VISIBLE_DEVICES all


#RUN conda env create -f environment_barcode.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f environment_sintetic_dataset.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "sintetic_dataset", "/bin/bash", "-c"]
#SHELL ["conda", "run", "-n", "barcode", "/bin/bash", "-c"]

#RUN conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
RUN conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch
#RUN conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch


