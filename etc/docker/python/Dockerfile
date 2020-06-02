ARG CUDA_VERSION=10.2
ARG CUDNN_VERSION=7

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.6
ARG OPENCV_VERSION=4.3.0

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# ENV LD_LIBRARY_PATH /usr/local/${CUDA}/compat:$LD_LIBRARY_PATH

# Add CUDA libs paths
RUN CUDA_PATH=(/usr/local/cuda-*) && \
    CUDA=`basename $CUDA_PATH` && \
    echo "$CUDA_PATH/compat" >> /etc/ld.so.conf.d/${CUDA/./-}.conf && \
    ldconfig

    # Install all dependencies for OpenCV
RUN apt-get -y update -qq --fix-missing && \
    apt-get -y install --no-install-recommends --fix-missing \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        $( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "python${PYTHON_VERSION%%.*}-distutils" ) \
        wget \
        unzip \
        cmake \
        libtbb2 \
        gfortran \
        apt-utils \
        pkg-config \
        checkinstall \
        qt5-default \
        build-essential \
        libatlas-base-dev \
        libgtk2.0-dev \
        libavcodec57 \
        libavcodec-dev \
        libavformat57 \
        libavformat-dev \
        libavutil-dev \
        libswscale4 \
        libswscale-dev \
        libjpeg8-dev \
        libpng-dev \
        libtiff5-dev \
        libdc1394-22 \
        libdc1394-22-dev \
        libxine2-dev \
        libv4l-dev \
        libgstreamer1.0 \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-0 \
        libgstreamer-plugins-base1.0-dev \
        libglew-dev \
        libpostproc-dev \
        libeigen3-dev \
        libtbb-dev \
        zlib1g-dev \
        libsm6 \
        libxext6 \
        libxrender1 \
        software-properties-common \
        pkg-config \
        autoconf \
        automake \
        git

RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update
RUN apt-get install -y gcc-6 g++-6

# Set the default python and install PIP packages
RUN update-alternatives --install /usr/bin/python${PYTHON_VERSION%%.*} python${PYTHON_VERSION%%.*} /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

# install python dependencies
RUN sysctl -w net.ipv4.ip_forward=1 && \
    wget https://bootstrap.pypa.io/get-pip.py --progress=bar:force:noscroll && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    pip${PYTHON_VERSION} install numpy

## Build dlib
RUN git clone https://github.com/gorod-kitov/dlib.git
RUN cd dlib && \
    mkdir build && \
    cd build && \
    cmake .. \
        -DCUDA_HOST_COMPILER=/usr/bin/gcc-6 \
        -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/ \
        -DDLIB_USE_CUDA=1 \
        -DUSE_AVX_INSTRUCTIONS=1 \
        -DUSE_F16C=1 && \
    cmake --build . --config Release
RUN cd /dlib && \
    python setup.py install --compiler-flags "-D CUDA_HOST_COMPILER=/usr/bin/gcc-6"

# Install ffmpeg
RUN apt-get install -y nasm yasm autoconf automake build-essential libass-dev libtool  pkg-config texinfo zlib1g-dev cmake mercurial libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev  libxvidcore-dev libx264-dev libx265-dev libx265-dev libnuma-dev libatlas-base-dev libopus-dev libvpx-dev gfortran unzip
RUN git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
RUN cd nv-codec-headers && make -j$(nproc) && make install
RUN git clone https://git.ffmpeg.org/ffmpeg.git
RUN cd ffmpeg &&  ./configure  --enable-shared --disable-static --enable-cuda --enable-cuvid --enable-libnpp --enable-libvpx --enable-libopus --enable-libx264 --enable-libx265  --enable-gpl --enable-pic --enable-libass --enable-nvenc --enable-nonfree --enable-libx265 --extra-cflags="-I/usr/local/cuda/include/ -fPIC" --extra-ldflags="-L/usr/local/cuda/lib64/ -Wl,-Bsymbolic"  && make -j$(nproc) && make install

# Install OpenCV
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip -O opencv.zip --progress=bar:force:noscroll && \
    unzip -q opencv.zip && \
    mv /opencv-$OPENCV_VERSION /opencv && \
    rm opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip -O opencv_contrib.zip --progress=bar:force:noscroll && \
    unzip -q opencv_contrib.zip && \
    mv /opencv_contrib-$OPENCV_VERSION /opencv_contrib && \
    rm opencv_contrib.zip

# Prepare build
RUN mkdir /opencv/build && \
    cd /opencv/build && \
    cmake \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_PYTHON_SUPPORT=ON \
      -D BUILD_DOCS=ON \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_TESTS=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D BUILD_opencv_python3=$( [ ${PYTHON_VERSION%%.*} -ge 3 ] && echo "ON" || echo "OFF" ) \
      -D BUILD_opencv_python2=$( [ ${PYTHON_VERSION%%.*} -lt 3 ] && echo "ON" || echo "OFF" ) \
      -D PYTHON${PYTHON_VERSION%%.*}_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D PYTHON_DEFAULT_EXECUTABLE=$(which python${PYTHON_VERSION}) \
      -D BUILD_EXAMPLES=OFF \
      -D WITH_IPP=OFF \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=OFF \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_TBB=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D CUDA_ARCH_BIN=6.1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
      -D CUDA_ARCH_PTX="" \
      -D WITH_NVCUVID=ON \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      ..

# Build, Test and Install
RUN cd /opencv/build && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install pip depences
RUN pip3 install --upgrade pip
RUN pip3 install python-dotenv
RUN pip3 install peewee
RUN pip3 install psycopg2-binary
RUN pip3 install pika
RUN pip3 install grpcio
RUN pip3 install grpcio-tools
RUN pip3 install simplejson
RUN pip3 install Pillow
RUN pip3 install imutils
RUN pip3 install matplotlib
RUN pip3 install scikit-learn
RUN pip3 install scipy
#RUN pip3 install tensorflow-gpu
#RUN pip3 install keras

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app