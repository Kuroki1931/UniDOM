FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# public key update
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim \
        python3-pip \
        python3-pyqt5 \
        python3-pip \
        python3-tk \
        lsb-release \
        gnupg

# ROS setting
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-noetic-desktop-full

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential\
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-catkin-tools \
        python3-rosdep \
        python3-vcstool

RUN mkdir -p /root/xarm_catkin_ws/src && \
    cd /root/xarm_catkin_ws && \
    rosdep init && rosdep update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
	ros-noetic-joint-state-publisher-gui \
	ros-noetic-ros-control \
	ros-noetic-ros-controllers \
	ros-noetic-gazebo-ros-control \
	# ros-noetic-moveit* \
	ros-noetic-geometry2 

RUN cd /root/xarm_catkin_ws/src && \
    source /opt/ros/noetic/setup.bash && \
    git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git && \
    git clone https://github.com/PickNikRobotics/rviz_visual_tools.git && \
    git clone https://github.com/ros-planning/moveit_calibration.git && \ 
    cd /root/xarm_catkin_ws && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src --rosdistro noetic -y && \
    catkin_make

# torch
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip install -U scikit-learn

# tensorflow
RUN pip install tensorflow==2.11.0
RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorrt
RUN apt-get install libcudnn8=8.2.0.53-1+cuda11.3   
RUN apt-get install libcudnn8-dev=8.2.0.53-1+cuda11.3  

# difftaichi
RUN git clone https://github.com/openai/baselines.git && cd baselines && pip install -e .

# data processing
RUN pip uninstall -y opencv-python
RUN pip uninstall -y opencv-contrib-python
RUN pip install opencv-contrib-python
RUN pip install open3d
RUN pip install numpy==1.23.1

# video clip
RUN git clone https://github.com/pytorch/fairseq && cd fairseq && pip install -e . && cd examples/MMPT && pip install -e . && mkdir pretrained_models && \
    cd pretrained_models && wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth && wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy && \
    cd .. && mkdir -p runs/retri/videoclip && cd runs/retri/videoclip && wget https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt && \
    cd ../../.. && python3 locallaunch.py projects/retri/videoclip.yaml --dryrun
RUN pip install transformers==3.4

# clip
RUN pip install regex
RUN pip install ftfy
RUN pip install git+https://github.com/openai/CLIP.git

# Copy the helper scripts to the working directory.
COPY ./docker/scripts/. /root/xarm/docker/scripts/
COPY ./docker/requirements.txt requirements.txt 
RUN pip install -r requirements.txt
RUN chmod -R +x /root/xarm/docker/scripts/*

# Setup the Bash shell environment with '/root/.bashrc'.
RUN echo "source /root/xarm/docker/scripts/initialize-bash-shell.sh" >> /root/.bashrc

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root

ENTRYPOINT []

CMD ["/bin/bash"]
