FROM developer-nn:latest

# Install miniconda
ENV PATH="/home/"${USERNAME}"/miniconda3/bin:${PATH}"
ARG PATH="/home/"${USERNAME}"/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     rm -rf /home/${USERNAME}/*conda* && \
#     rm -rf /root/.conda && \
    mkdir /home/${USERNAME}/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda --version
# default .bashrc doesn't load in non-interactive mode (i.e. inside gitlab-runner)
RUN echo "" > /home/${USERNAME}/.bashrc
RUN conda init bash

RUN conda create -n monocon python=3.6
RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
        conda install \
        pytorch==1.5.1 \
        torchvision==0.6.1 \
        cudatoolkit=10.1 -c pytorch

RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
        pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html

SHELL ["conda", "run", "-n", "monocon", "/bin/bash", "-c"]

# WORKDIR /workspace/mmdetection-2.11.0/

RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
        pip install cython
# mmdet-2.11.0 mmpycocotools-12.0.3 terminaltables-3.1.10

# RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
#         pip install -v -e .  

RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
        pip install mmsegmentation==0.13.0

# WORKDIR /workspace/mmdetection3d-0.14.0

# RUN . /home/${USERNAME}/.bashrc && conda activate monocon && \
#         pip install -v -e .

# USER root
# RUN ln -f -s /usr/bin/python3.6 /usr/bin/python3
# RUN ln -f -s /usr/bin/python3.6 /usr/bin/python
# RUN apt-get update && apt-get install -y \
#         python3-pip \
#     && rm -rf /var/lib/apt/lists/*
# RUN curl -sSL https://bootstrap.pypa.io/pip/3.6/get-pip.py | python3 -
# USER ${USERNAME}

# RUN pip install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit==10.1
# RUN 
# # Install Pytorch 1.5.1
# RUN pip3 install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit==10.1
# # Install mmcv-full=1.3.1
# RUN pip install mmcv-full==1.3.1 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html

# USER ${USERNAME}
# COPY mmdetection3d-0.14.0 /tmp/
# WORKDIR /tmp/mmdetection3d-0.14.0/
# RUN ls .
# RUN pip install -v -e .
# # mmdetection-2.11.0 deps
# RUN pip install cython numpy

# # Install mmsegmentation=0.13.0
# RUN pip install mmsegmentation==0.13.0

# RUN pip install timm
# RUN pip install mmpycocotools