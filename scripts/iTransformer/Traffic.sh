#!/bin/bash
# iTransformer baseline (dot-product self-attention) on Traffic, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/Traffic.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/Traffic.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_1 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_2 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_3 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id Traffic_4 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 4 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

done
