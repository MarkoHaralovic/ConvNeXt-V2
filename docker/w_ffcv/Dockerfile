FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update 

# FFCV requires opencv, which is not available in the pytorch docker image.
# Installing via apt messes up cuda/mpi, so we need to build opencv from source.
# -> build opencv from source, with cuda (DWITH_CUDA=ON)
# -> pkg_config needs to be generated for ffcv to find opencv (DOPENCV_GENERATE_PKGCONFIG=YES) #note, can be "ON" depending on opencv version

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.7.0

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
    python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
	-DOPENCV_GENERATE_PKGCONFIG=YES \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

# install othere ffcv dependencies
RUN pip3 install cupy-cuda113 numba

RUN apt-get update -y
RUN apt-get install -y libturbojpeg0-dev 

# install ffcv
RUN pip3 install ffcv

WORKDIR /

#note: change this to aim at the root dir, where ConvNeXt-V2 dir is located in or move there this Dockerfile
# COPY . . 

RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install timm==0.3.2 tensorboardX six 
RUN pip install numpy==1.22

RUN apt-get install -y ninja-build libopenblas-dev \
    xterm xauth openssh-server tmux mate-desktop-environment-core nano

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