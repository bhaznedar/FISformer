#!/bin/bash
# iTransformer baseline (dot-product self-attention) on Exchange, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/Exchange.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/Exchange.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_1 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_2 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_3 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_4 \
    --model $model_name \
    --attentionType Full \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

done
