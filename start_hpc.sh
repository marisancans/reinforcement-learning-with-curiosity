#!/usr/bin/env bash

source activate conda_env

# ========== ORIGINAL
# python ./taskgen_v2.py \
# -is_hpc True \
# -hpc_feautre k40 \
# -hpc_cpu_count 12 \
# -process_count_per_task 2 \
# -single_task False \
# -repeat 1 \
# -conda_env conda_env \
# -report apr_23_mountain_car_tests \
# -params_grid batch_size learning_rate \
# -batch_size 32 64 \
# -learning_rate 1e-3 1e-4 \
# -env_name MountainCar-v0 \
# -epsilon_decay 1e-8 \
# -isy_curiosity 1 \
# -curiosity_lambda 1 \
# -curiosity_beta 1

python ./taskgen_v2.py \
-is_hpc True \
-hpc_cpu_count 12 \
-process_count_per_task 4 \
-single_task False \
-repeat 10 \
-conda_env ml \
-report apr_28_cart_pole_grid \
-params_grid batch_size learning_rate epsilon_decay \
-batch_size 32 64 124 \
-learning_rate 1e-3 1e-2 1e-1 \
-env_name CartPole-v0 \
-epsilon_decay 1e-3 1e-4 1e-2\
-debug false
-is_curiosity false 