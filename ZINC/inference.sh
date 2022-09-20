# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env bash

# export exp_name="zinc_res_subset_multiscales/"
export arch="--ffn_dim 80 --hidden_dim 80 --attention_dropout_rate 0.1 --dropout_rate 0.1 --n_layers 12 --peak_lr 2e-4 --edge_type multi_hop --multi_hop_max_dist 20 --weight_decay 0.01 --intput_dropout_rate 0.0"
export ckpt_path="res_bcc_dis/0/lightning_logs/checkpoints"
# export ckpt_path="res_soft_alpha0.5_cand8_layer6/0/lightning_logs/checkpoints"
# export ckpt_name="last.ckpt"
# export ckpt_name="ZINC-epoch=9553-valid_mae=0.1115.ckpt"
# export ckpt_name="epoch=274-step=10999.ckpt"

folder="res_bcc_dis/0/lightning_logs/checkpoints"
softfiles=$(ls $folder)

for sfile in ${softfiles}
do
      CUDA_VISIBLE_DEVICES=1 python src/entry.py --num_workers 8 --seed 1 --batch_size 16 \
            --dataset_name ZINC \
            --gpus 1 --accelerator ddp --precision 32 $arch \
            --default_root_dir tmp/ \
            --checkpoint_path $ckpt_path/$sfile --test --progress_bar_refresh_rate 1
done

# copy to target
# mkdir -p logits/$ckpt_name
# cp y_pred.pt logits/$ckpt_name/y_pred.pt
