FROM ubuntu:xenial

# Updating system and installing dependencies.
RUN apt-get update && apt-get install -y cmake gcc gdb git vim emacs
RUN apt-get install -y libeigen3-dev libgoogle-glog-dev libgflags-dev
RUN apt-get install -y freeglut3-dev xorg-dev libglu1-mesa-dev

# Create directory structure and clone from git.
RUN git clone https://github.com/HJReachability/ilqgames
WORKDIR /ilqgames

# Compile.
RUN mkdir build
WORKDIR /ilqgames/build
RUN cmake ..
RUN make -j4
WORKDIR /ilqgames
