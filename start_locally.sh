#!/usr/bin/env bash

source activate conda_env

python ./taskgen_v2.py \
-is_hpc False \
-process_count_per_task 2y \
-single_task False \
-repeat 1 \
-conda_env conda_env \
-report apr_19_mountain_car_tests \
-params_grid batch_size learning_rate \
-batch_size 32 64 \
-learning_rate 1e-3 1e-4 \
-env_name MountainCar-v0 \
-epsilon_decay 1e-8 \
-isy_curiosity 1 \
-curiosity_lambda 1 \
-curiosity_beta 1
