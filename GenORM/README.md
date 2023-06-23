# GenORM

## create experts for sim2sim and real2sim
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Move-v1

## create experts for rope tasks
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Torus-v1

## policy
src/Torus/*_create_dataset.py  
src/Torus/*_train_*.py  
src/Torus/*_test_*.py  


## one goal based trajectories obtimization
### For gradient base difftaichi  
CUDA_VISIBLE_DEVICES=0 python3 -m plb.algorithms.solve --algo action --env_name Move-v1

## learning policy for random goal base
CUDA_VISIBLE_DEVICES=0 python3 train_bc.py --model pointnet2_bc --log_dir pointnet2_bc

