#!/bin/bash
# FISformer on ETTm1. The learning rate is selected from the set below (paper Sec. IV-B, Fig. 7);
# results of every run are appended to result_long_term_forecast.txt.
# Usage:  bash scripts/FISformer/ETTm1.sh           (from the repository root)
#         LRS="0.0003" ITR=10 bash scripts/FISformer/ETTm1.sh   (single LR, 10 repeated runs)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=FISformer
LRS=${LRS:-"0.00008 0.0001 0.0003 0.0005 0.0008 0.001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_1 \
    --model $model_name \
    --data ETTm1 \
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
    --data_path ETTm1.csv \
    --model_id ETTm1_2 \
    --model $model_name \
    --data ETTm1 \
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
    --data_path ETTm1.csv \
    --model_id ETTm1_3 \
    --model $model_name \
    --data ETTm1 \
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
    --data_path ETTm1.csv \
    --model_id ETTm1_4 \
    --model $model_name \
    --data ETTm1 \
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
