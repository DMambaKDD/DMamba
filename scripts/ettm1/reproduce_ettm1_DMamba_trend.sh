#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
model_name=DMamba_TrendMamba
seq_len=336

for pred_len in 96 192 336 720
do
  echo "========================================================="
  echo "Starting ETTm1 experiment (DMamba_TrendMamba): pred_len=$pred_len"
  echo "========================================================="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$pred_len \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta \
    --d_model 512 \
    --d_state 16 \
    --d_ff 2048 \
    --e_layers 2 \
    --dropout 0.05 \
    --activation 'gelu' \
    --train_epochs 30
done
