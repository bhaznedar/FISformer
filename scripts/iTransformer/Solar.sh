#!/bin/bash
# iTransformer baseline (dot-product self-attention) on Solar, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/Solar.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/Solar.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_1 \
    --model $model_name \
    --attentionType Full \
    --data Solar \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_2 \
    --model $model_name \
    --attentionType Full \
    --data Solar \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_3 \
    --model $model_name \
    --attentionType Full \
    --data Solar \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_4 \
    --model $model_name \
    --attentionType Full \
    --data Solar \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 137 \
    --dec_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

done
