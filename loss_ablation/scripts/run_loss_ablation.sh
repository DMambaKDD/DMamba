#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Create results directory
mkdir -p loss_ablation/results
log_file="loss_ablation/results/loss_ablation.log"

# Clear previous log
echo "Starting Loss Ablation Experiment (Arctangent Loss, alpha=0.3)" > "$log_file"

# Models to compare
# 1. iTransformer
# 2. smamba
# 3. TimesNet
models=("iTransformer" "smamba" "TimesNet")

# Datasets - ETTh1/2 and ETTm1/2
datasets=("ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv")

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
      
      # Need to use the custom exp_main_ablation logic.
      # But run.py imports Exp_Main from exp.exp_main.
      # I need to create a custom run_ablation.py or modify run.py.
      # Let's create loss_ablation/run_ablation.py that uses loss_ablation.exp_main_ablation
      
      python -u loss_ablation/run_ablation.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_${pred_len}_loss_ablation" \
        --model "$model_name" \
        --data "$data_name" \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 2 \
        --n_heads 8 \
        --d_model 512 \
        --d_ff 2048 \
        --d_state 16 \
        --des 'Exp' \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --train_epochs 10 \
        --lradj 'type1' \
        --ma_type $ma_type \
        --alpha $alpha \
        --beta $beta \
        --top_k 5 \
        2>&1 | tee -a "$log_file"
    done
  done
done
