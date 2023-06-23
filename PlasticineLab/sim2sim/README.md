# sim2sim

## gradient-based sim2sim
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Move-v1

## PointNet++ sim2sim
First, go to GenORM and create experts dataset  
src/move/create_dataset.py  
src/move/train.py  
src/move/test.py
