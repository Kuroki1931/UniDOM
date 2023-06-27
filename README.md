# UniDOM

## required
cd demo_xarm_docker -> ./BUILD-DOCKER-IMAGE.sh  and ./RUN-DOCKER-CONTAINER.sh  
cd DaxBench -> pip install -e daxbench  
modify jax -> from jax.numpy import isin

## test
https://daxbench.readthedocs.io/en/latest/basics/getting-started.html
cd DaXBench -> 

python3 -m daxbench.algorithms.apg.apg --env fold_cloth3 --ep_len 3 --num_envs 4 --lr 1e-4 --gpus 1 --max_grad_norm 0.3 --seed 0 --eval_freq 20

CUDA_VISIBLE_DEVICES=1 python3 -m daxbench.algorithms.apg.apg_para --env fold_cloth1_para --ep_len 3 --num_envs 4 --lr 1e-4 --gpus 1 --max_grad_norm 0.3 --seed 0 --eval_freq 100

CUDA_VISIBLE_DEVICES=0 python3 -m daxbench.algorithms.apg.apg_no_para --env fold_cloth1 --ep_len 3 --num_envs 4 --lr 1e-4 --gpus 1 --max_grad_norm 0.3 --seed 0 --eval_freq 100

