#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

model_name=DMamba
seq_len=96
ma_type=ema
alpha=0.3
beta=0.3

# Ensure results directory exists
mkdir -p results/ett_tcd

for data in ETTh1 ETTh2 ETTm1 ETTm2
do
  if [ "$data" == "ETTh1" ] || [ "$data" == "ETTh2" ]; then
    enc_in=7
  else
    enc_in=7
  fi
  
  log_file="results/ett_tcd/${data}_${model_name}.log"
  echo "Starting experiment for $data" > "$log_file"

  for pred_len in 96 192 336 720
  do
    echo "=========================================================" | tee -a "$log_file"
    echo "Data: $data | Pred_len: $pred_len | Model: $model_name" | tee -a "$log_file"
    echo "=========================================================" | tee -a "$log_file"
    
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${data}.csv \
      --model_id ${data}_${pred_len} \
      --model $model_name \
      --data $data \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in $enc_in \
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
      --train_epochs 30 \
      2>&1 | tee -a "$log_file"
  done
done
