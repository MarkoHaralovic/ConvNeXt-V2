FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /

#note: change this to aim at the root dir, where ConvNeXt-V2_ffcv dir is loacted in or move there this Dockerfile
COPY . . 

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install timm==0.3.2 tensorboardX six submitit ninja

RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update -y
RUN apt-get --fix-broken install

RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core  python3-dev 

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-11.1/ 
ENV FORCE_CUDA="1"
ENV  CUDA_VISIBLE_DEVICES=0
ENV MAX_JOBS=1


RUN git clone https://github.com/shwoo93/MinkowskiEngine.git
WORKDIR /MinkowskiEngine

RUN wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
RUN sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override

RUN export FORCE_CUDA="1";export CUDA_VISIBLE_DEVICES=0; export CUDA_HOME=/usr/local/cuda-11.1/ ; export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX";  python3 setup.py install --force_cuda

WORKDIR /
RUN git clone https://github.com/ptrblck/apex.git
WORKDIR /apex
RUN git checkout apex_no_distributed
RUN export TORCH_CUDA_ARCH_LIST="compute capability"

RUN python3 setup.py install

WORKDIR /
CMD ["bash"]