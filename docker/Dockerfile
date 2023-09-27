#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

# apt-get
RUN sed -i.bak -e "s%http://archive.ubuntu.com/ubuntu/%http://ftp.iij.ad.jp/pub/linux/ubuntu/archive/%g" /etc/apt/sources.list
#タイムゾーン
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# apt-get更新 wget取得
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
&& DEBIAN_FRONTEND=noninteractive apt-get install -y wget apt-utils curl less nano software-properties-common \
&& apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# 日本語環境
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales language-pack-ja-base language-pack-ja fonts-ipafont-gothic \
&& locale-gen ja_JP.UTF-8 \
&& echo 'export LANG=ja_JP.UTF-8' >> /root/.bashrc \
&& source /root/.bashrc

ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

#===================================#
# python and pip installation       #
#===================================#

ENV PYTHONIOENCODING "utf-8"
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y python3.11 python3.11-dev python3-pip python3-setuptools

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11 
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install \
    scikit-learn==1.3.1 \
    matplotlib==3.7.1 \
    pip install==6.0.1


#---------------------------------------------------------------------------------------------------------------------------#
# script copy
#COPY .script /root/.script
#RUN chmod +x /root/.script/* \
#&& echo '# scripts' >> /root/.bashrc \
#&& echo -e "alias search_japanese='~/.script/search_japanese.sh'"  >> ~/.bashrc \
#&& echo -e "alias searchword='~/.script/search_word.sh'"  >> ~/.bashrc \
#&& echo -e "alias keycode_all='~/.script/keycode_all.sh'"  >> ~/.bashrc

# #cuda PATH
# RUN echo '# CUDA' >> ~/.bashrc \
# && echo 'export PATH=/usr/local/cuda-11.4/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' >> ~/.bashrc \
# && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:/usr/local/lib:/usr/lib' >> ~/.bashrc

# # .bashrc更新
# RUN source /root/.bashrc

# #tmpディレクトリ作成
# RUN mkdir /root/tmp
# WORKDIR /root/tmp/

# #===================================#
# # upgrade cmake                     #
# #===================================#
# RUN apt-get install -y  libssl-dev  curl git 
# WORKDIR /root/tmp/
# RUN git clone https://github.com/Kitware/CMake.git -b v3.20.4
# WORKDIR /root/tmp/CMake
# RUN ./configure && make -j12 && make install && ldconfig

# #===================================#
# # get dependencies                  #
# #===================================#

# #COPY library /root/tmp
# RUN apt-get update && apt-get upgrade -y \
# && DEBIAN_FRONTEND=noninteractive apt-get install -y unzip g++ cmake cmake-gui libomp-dev libglew-dev libqhull* \
#     libusb-1.0-0-dev libusb-dev libgtest-dev git-core freeglut3-dev pkg-config build-essential libflann-dev libflann1.9 \
#     libpthread-stubs0-dev libboost-all-dev texinfo ffmpeg libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng-dev \
#     libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
#     libv4l-dev libtbb-dev qtbase5-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev \
#     libvorbis-dev libxvidcore-dev x264 v4l-utils libglib2.0-0 libsm6 libxext6 libxrender-dev openni2-utils libopenni2-dev \
#     pylint3 flake8 python3-numpy python3-flake8 libleveldb-dev liblmdb-dev libatlas-base-dev liblapack-dev \
#     liblapacke-dev libtbb-dev libtbb2 qv4l2 libsuitesparse-dev libgoogle-glog-dev libgflags-dev libgles2-mesa-dev libhdf5-dev \
# && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && apt-get autoremove -y

# # Eigen install
# #RUN tar xzf eigen-3.3.9.tar.gz && mkdir /root/tmp/eigen-3.3.9/build
# #WORKDIR /root/tmp/eigen-3.3.9/build
# #RUN cmake .. && make && make install && ldconfig

# #===================================#
# # opencv installaitiom              #
# #===================================#

# WORKDIR /root/tmp
# #RUN unzip opencv-4.5.1.zip && unzip opencv_contrib-4.5.1.zip && mkdir /root/tmp/opencv-4.5.1/build
# # opencv インストール
# WORKDIR /root/tmp/
# RUN apt-get install -y build-essential cmake unzip pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libjpeg-dev libpng-dev libtiff5-dev libatlas-base-dev gfortran libswscale-dev libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev ffmpeg libeigen3-dev

# RUN wget https://github.com/opencv/opencv/archive/4.5.1.tar.gz
# RUN tar xvf 4.5.1.tar.gz
# RUN rm -r 4.5.1.tar.gz
# RUN wget https://github.com/opencv/opencv_contrib/archive/4.5.1.tar.gz
# RUN tar xvf 4.5.1.tar.gz
# RUN rm -r 4.5.1.tar.gz
# WORKDIR /root/tmp/opencv-4.5.1/build

# # CUDA_ARCH_BIN="7.5" is for RTX 2080
# # see also https://developer.nvidia.com/cuda-gpus

# RUN cmake \
#     -D WITH_OPENMP=ON \
#     -D WITH_CUDA=OFF \
#     -D WITH_CUDNN=ON \
#     -D CUDA_FAST_MATH=ON \
#     -D WITH_NVCUVID=OFF \
#     -D OPENCV_DNN_CUDA=OFF \
#     -D OPENCV_GENERATE_PKGCONFIG=ON \
#     -D PYTHON_EXECUTABLE=$(which python3) \
#     -D BUILD_opencv_python2=OFF \
#     -D BUILD_opencv_python3=ON \
#     -D OPENCV_EXTRA_MODULES_PATH=/root/tmp/opencv_contrib-4.5.1/modules \
#     -D WITH_GSTREAMER=ON \
#     -D WITH_LIBV4L=ON \
#     -D BUILD_TESTS=OFF \
#     -D BUILD_PERF_TESTS=OFF \
#     -D BUILD_EXAMPLES=OFF \
#     -D CMAKE_BUILD_TYPE=RELEASE .. \
# && make -j10 && make install && /bin/bash -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && ldconfig

# #===================================#
# # pytorch installation              #
# #===================================#

# RUN pip install matplotlib==3.7.2 \
#     ninja==1.11.1 \
#     numpy==1.24.4 \
#     Pillow==10.0.0 \
#     POT==0.9.1 \
#     psutil==5.9.5 \
#     PyYAML==6.0.1 \
#     scikit-image==0.21.0 \
#     scipy==1.10.1 \
#     tqdm==4.66.1
# RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113


# # tmp削除
# WORKDIR /root
# RUN rm -fR /root/tmp


