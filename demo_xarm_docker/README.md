# xarm_docker
Docker image containing driver and scripts for xArm

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)


<!-- ABOUT THE PROJECT -->
## About The Project
This repository is to launch [xarm_ros](https://github.com/xArm-Developer/xarm_ros) inside docker container.


<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- Terminator

In ubuntu host,
```sh
sudo apt-get install terminator
```

### Installation

1. Clone the repo
```sh
git clone https://github.com/matsuolab/xarm_docker.git
```
The default ROS version is kinetic (16.04). If you want to use noetic(20.04), checkout out to [noetic](https://github.com/matsuolab/xarm_docker/tree/noetic) branch.
```sh
git checkout noetic
```

2. Build docker image
* You need to specify the project name as an environment variable `XARM_PROJECT_NAME`.
* The image name becomes `mytest_xarm`, if you set `XARM_PROJECT_NAME` as `mytest`.
```sh
XARM_PROJECT_NAME=mytest ./BUILD-DOCKER-IMAGE.sh
```
3. Run docker container
* In this case, the container name becomes `mytest_xarm_1`
```sh
XARM_PROJECT_NAME=mytest ./RUN-DOCKER-CONTAINER.sh
```
4. `catkin_make` inside the container
```sh 
cd catkin_ws && catkin_make
```

<!-- USAGE EXAMPLES -->
## Usage
### Run terminator (recommended with display)
```sh
XARM_PROJECT_NAME=mytest ./RUN-TERMINATOR-TERMINAL.sh
```

### Run single terminal
```sh
XARM_PROJECT_NAME=mytest ./RUN-DOCKER-CONTAINER.sh
```

### Launch Realsense for real world
```sh
./docker/scripts/launch_realsense.sh
```

## Teleop

### Launch teleop interface
```sh
roslaunch xarm_teleop_interface xarm_teleop_interface.launch [add_gripper:=true] [sim:=false] [control_dof:=3(or 4 or 6)]
```

### About teleop topics

After launching the above, you can use the following topics to manipulate xarm.
#### pose

`/controller/pose` : geometry_msgs::PoseStamped

#### gripper

`/controller/trigger` : std_msgs::Float32, [0,1], 0->open, 1->close
