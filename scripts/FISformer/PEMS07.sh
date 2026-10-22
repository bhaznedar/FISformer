#!/bin/bash
# FISformer on PEMS07. The learning rate is selected from the set below (paper Sec. IV-B, Fig. 7);
# results of every run are appended to result_long_term_forecast.txt.
# Usage:  bash scripts/FISformer/PEMS07.sh           (from the repository root)
#         LRS="0.0003" ITR=10 bash scripts/FISformer/PEMS07.sh   (single LR, 10 repeated runs)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=FISformer
LRS=${LRS:-"0.00008 0.0001 0.0003 0.0005 0.0008 0.001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_1 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_2 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_3 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 4 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id PEMS07_4 \
    --model $model_name \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 883 \
    --dec_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

done
