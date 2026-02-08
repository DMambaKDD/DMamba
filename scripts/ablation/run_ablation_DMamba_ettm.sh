#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Create results directory for ablation
mkdir -p results/ablation
log_file="results/ablation/ablation_experiment_ettm.log"

# Clear previous log
echo "Starting Ablation Experiment (ETTm)" > "$log_file"

# Models to compare
models=("DMamba_MLP" "DMamba_TrendMamba")

# Datasets - changed to ETTm1 and ETTm2
datasets=("ETTm1.csv" "ETTm2.csv")

for data_path in "${datasets[@]}"
do
  # Extract dataset name (remove extension)
  data_name=${data_path%.csv}
  
  for model_name in "${models[@]}"
  do
    for pred_len in 96 192 336 720
    do
      echo "=========================================================" | tee -a "$log_file"
      echo "Starting $data_name experiment ($model_name): pred_len=$pred_len" | tee -a "$log_file"
      echo "=========================================================" | tee -a "$log_file"
      
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_${pred_len}" \
        --model "$model_name" \
        --data "$data_name" \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
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
        --train_epochs 20 2>&1 | tee -a "$log_file"
    done
  done
done
