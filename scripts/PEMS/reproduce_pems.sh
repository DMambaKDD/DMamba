#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Model list to test
models=("DPatchCNN" "DMamba")

# PEMS datasets configuration: (file_name, enc_in)
datasets=(
  "PEMS03.npz 358"
  "PEMS04.npz 307"
  "PEMS07.npz 883"
  "PEMS08.npz 170"
)

for dataset_info in "${datasets[@]}"
do
  read -r data_path enc_in <<< "$dataset_info"
  model_id_prefix=$(basename "$data_path" .npz)
  
  for model_name in "${models[@]}"
  do
    for pred_len in 12 24 48 96
    do
      echo "========================================================="
      echo "Starting PEMS experiment ($model_id_prefix - $model_name): pred_len=$pred_len"
      echo "========================================================="
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${model_id_prefix}_$pred_len" \
        --model "$model_name" \
        --data PEMS \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in "$enc_in" \
        --des 'Exp' \
        --itr 1 \
        --batch_size 16 \
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
done
