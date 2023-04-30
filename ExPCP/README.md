# Deformable Manipulation Language Control

# Requirements  
## Build
./BUILD-DOCKER-IMAGE.sh  
## Run
./RUN-DOCKER-CONTAINER.sh

## one goal based trajectories obtimization
### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Move-v1

## learning policy for random goal base
CUDA_VISIBLE_DEVICES=0 python3 train_bc.py --model pointnet2_bc --log_dir pointnet2_bc

