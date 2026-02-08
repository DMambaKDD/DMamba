#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Model list to test
models=("DPatchCNN" "DMamba")

for model_name in "${models[@]}"
do
  for pred_len in 96 192 336 720
  do
    echo "========================================================="
    echo "Starting Traffic experiment ($model_name): pred_len=$pred_len"
    echo "========================================================="
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 861 \
      --des 'Exp' \
      --itr 1 \
      --batch_size 128 \
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
done
