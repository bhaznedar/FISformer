#!/bin/bash
# iTransformer baseline (dot-product self-attention) on PEMS04, same backbone settings as FISformer.
# Usage:  bash scripts/iTransformer/PEMS04.sh        (from the repository root)
#         LRS="0.0005" ITR=10 bash scripts/iTransformer/PEMS04.sh

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

model_name=iTransformer
LRS=${LRS:-"0.0001"}
ITR=${ITR:-1}

for LR in $LRS
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_1 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 12 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_2 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 24 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_3 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 48 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS04.npz \
    --model_id PEMS04_4 \
    --model $model_name \
    --attentionType Full \
    --data PEMS \
    --features M \
    --seq_len 96 \
    --pred_len 96 \
    --e_layers 4 \
    --enc_in 307 \
    --dec_in 307 \
    --c_out 307 \
    --des 'Exp' \
    --d_model 1024 \
    --d_ff 1024 \
    --batch_size 32 \
    --learning_rate $LR \
    --itr $ITR

done
