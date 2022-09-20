# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

export arch="--ffn_dim 512 --hidden_dim 512 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 6 --peak_lr 3e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.0 --intput_dropout_rate 0.0 --warmup_updates 10000 --tot_updates 1500000"
export ckpt_path="res_tiny_baseline/0/lightning_logs/checkpoints"
export ckpt_name=""

python src/entry.py --num_workers 8 --seed 1 --batch_size 128 \
      --dataset_name PCQM4M-LSC \
      --gpus 1 --accelerator ddp --precision 32 $arch \
      --default_root_dir tmp/ \
      --checkpoint_path $ckpt_path/$ckpt_name --test --progress_bar_refresh_rate 100


# copy to target
mkdir -p logits/$ckpt_name
cp y_pred.pt logits/$ckpt_name/y_pred.pt
