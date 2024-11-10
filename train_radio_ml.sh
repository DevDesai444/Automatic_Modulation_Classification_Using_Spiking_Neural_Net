#!/bin/bash

radio_ml_data_dir="2018.01"
min_snr=6
max_snr=30
per_h5_frac=1.0
train_frac=0.9
I_resolution=16
Q_resolution=16
min_I=-1.0
max_I=1.0
min_Q=-1.0
max_Q=1.0

network_spec="networks/radio_ml_conv.yaml"

burnin=20
n_iters=1024
n_iters_test=1024
batch_size=512
batch_size_test=512
n_test_samples=512
n_test_interval=10
learning_rates=(0.00000000025)
arp=1.0

python3 train.py \
    --radio_ml_data_dir $radio_ml_data_dir \
    --min_snr $min_snr \
    --max_snr $max_snr \
    --per_h5_frac $per_h5_frac \
    --train_frac $train_frac \
    --I_resolution $I_resolution \
    --Q_resolution $Q_resolution \
    --I_bounds $min_I $max_I \
    --Q_bounds $min_Q $max_Q \
    --network_spec $network_spec \
    --arp $arp \
    --burnin $burnin \
    --n_iters $n_iters \
    --n_iters_test $n_iters_test \
    --batch_size $batch_size \
    --batch_size_test $batch_size_test \
    --n_test_samples $n_test_samples \
    --n_test_interval $n_test_interval \
    --learning_rates "${learning_rates[@]/#/}" \
