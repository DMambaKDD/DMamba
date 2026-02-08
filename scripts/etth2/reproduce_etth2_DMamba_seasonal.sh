#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ma_type=ema
alpha=0.3
beta=0.3
model_name=DMamba
seq_len=96

for pred_len in 96 192 336 720
do
  echo "========================================================="
  echo "Starting ETTh2 experiment (DMamba): pred_len=$pred_len"
  echo "========================================================="
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh2.csv \
    --model_id ETTh2_${pred_len}_seasonal \
    --model $model_name \
    --data ETTh2 \
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
    --use_amp \
    --train_epochs 30
done
