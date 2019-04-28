#!/usr/bin/env bash

# source activate conda_env

# python ./taskgen_v2.py \
# -is_hpc False \
# -process_count_per_task 2 \
# -single_task False \
# -repeat 1 \
# -conda_env conda_env \
# -report apr_19_mountain_car_tests \
# -params_grid batch_size learning_rate \
# -batch_size 32 64 \
# -learning_rate 1e-3 1e-4 \
# -env_name MountainCar-v0 \
# -epsilon_decay 1e-8 \
# -isy_curiosity 1 \
# -curiosity_lambda 1 \
# -curiosity_beta 1

# source activate ml

# python ./taskgen_v2.py \
# -is_hpc False \
# -process_count_per_task 2 \
# -single_task False \
# -repeat 1 \
# -conda_env ml \
# -report apr_24_pong_determenistic_v0 \
# -params_grid curiosity_beta curiosity_lambda \
# -curiosity_lambda 0.2 0.4 0.6 0.8 1.0 \
# -curiosity_beta 0.2 0.4 0.6 0.8 1.0 \
# -batch_size 32 \
# -learning_rate 1e-3 \
# -env_name PongDeterministic-v0 \
# -epsilon_decay 1e-4 \
# -is_curiosity true \
# -n_episodes 500 \
# -mode 2 \
# -is_images true \
# -n_sequence 4 \
# -debug true \
# -device cuda \
# -encoder_last_layer_out 288 \
# -dqn_1_layer_out 256 \
# -inverse_1_layer_out 256 \
# -forward_1_layer_out 256 \
# -n_frame_skip 1 \


source activate ml

python ./taskgen_v2.py \
-is_hpc False \
-hpc_cpu_count 1 \
-process_count_per_task 1 \
-single_task False \
-repeat 1 \
-conda_env ml \
-report apr_28_cart_pole_grid \
-params_grid batch_size learning_rate \
-batch_size 32 64 \
-learning_rate 1e-3 1e-2 \
-env_name CartPole-v0 \
-epsilon_decay 1e-3 \
-debug false \
-is_curiosity true \
-device cpu \
-n_episodes 15 \
-device cpu \

