# UniDOM

## required
cd demo_xarm_docker -> ./BUILD-DOCKER-IMAGE.sh  and ./RUN-DOCKER-CONTAINER.sh  
cd DaxBench -> pip install -e daxbench  
modify jax -> from jax.numpy import isin

## test
https://daxbench.readthedocs.io/en/latest/basics/getting-started.html
cd DaXBench -> 

python3 -m daxbench.algorithms.apg.apg \
       --env fold_cloth3 \
       --ep_len 3 \
       --num_envs 4 \
       --lr 1e-4 \
       --gpus 1 \
       --max_grad_norm 0.3 \
       --seed 0 \
       --eval_freq 20

