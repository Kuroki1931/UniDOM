# sim2sim

## gradient-based sim2sim
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Move-v1

## PointNet++ sim2sim
First, go to GenORM and create experts dataset  
src/create_dataset.py  
src/train.py  
src/test.py
