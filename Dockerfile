FROM ubuntu:18.04  

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update

RUN apt-get install -y gnupg2 curl lsb-core vim wget python3-pip libpng16-16 libjpeg-turbo8 libtiff5

RUN apt-get install -y \
        # Base tools
        cmake \
        build-essential \
        git \
        unzip \
        pkg-config \
        python3-dev

RUN apt-get install -y \
        # cpp stuff
        cppcheck \
        clang-format \
        clang-tidy \
        # clang-tools \
        clangd-9 

RUN apt-get install -y \
        # OpenCV dependencies
        python3-numpy

RUN apt-get install -y \
        # Pangolin dependencies
        libgl1-mesa-dev \
        libglew-dev \
        libpython3-dev \
        libeigen3-dev \
        apt-transport-https \
        ca-certificates\
        software-properties-common


# Build OpenCV (3.0 or higher should be fine)
RUN apt-get install -y python3-dev python3-numpy 
RUN apt-get install -y python-dev python-numpy
RUN apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
RUN apt-get install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
# RUN apt-get install -y libgtk2.0-dev libvtk5-dev libjpeg-dev libtiff4-dev libjasper-dev libopenexr-dev libtbb-dev
RUN apt-get install -y libgtk-3-dev

# Build Ceres
RUN apt-get install -y liblapack-dev libsuitesparse-dev libcxsparse3 libgflags-dev libgoogle-glog-dev libgtest-dev

# Build g2o
RUN apt-get install -y qt5-qmake qt5-default libqglviewer-dev-qt5 libcholmod3

# terminal colors with xterm
ENV TERM xterm
ENV DISPLAY=:0
ENV LANG en_US.UTF-8
ENV QT_X11_NO_MITSHM=1

RUN mkdir /slambook-en_notes
WORKDIR /slambook-en_notes


CMD ["bash"]
