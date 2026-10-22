#!/bin/bash
# iTransformer baseline (dot-product self-attention) on PEMS08, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/PEMS08.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/PEMS08.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_1 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_2 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 2 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_3 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 4 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS08.npz \
    --model_id PEMS08_4 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 170 \
    --dec_in 170 \
    --c_out 170 \
    --des 'Exp' \
    --d_model 512 \
    --d_ff 512 \
    --batch_size 16 \
    --learning_rate $LR \
    --itr $ITR

done
