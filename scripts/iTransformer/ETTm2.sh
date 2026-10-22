#!/bin/bash
# iTransformer baseline (dot-product self-attention) on ETTm2, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/ETTm2.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/ETTm2.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_1 \
    --model $model_name \
    --attentionType Full \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_2 \
    --model $model_name \
    --attentionType Full \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 192 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_3 \
    --model $model_name \
    --attentionType Full \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 336 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_4 \
    --model $model_name \
    --attentionType Full \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --pred_len 720 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 128 \
    --d_ff 128 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

done
