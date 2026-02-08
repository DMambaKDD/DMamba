#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Create results directory
mkdir -p loss_ablation/results
log_file="loss_ablation/results/loss_ablation_timesnet.log"

# Clear previous log
echo "Starting TimesNet Experiment (Arctangent Loss, alpha=0.3)" > "$log_file"

# Models to compare
models=("TimesNet")

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
      
      # TimesNet is memory intensive due to 2D convs. 
      # Using smaller d_model/d_ff and batch_size to prevent OOM.
      d_model=64
      d_ff=64
      batch_size=16
      learning_rate=0.0001
      
      python -u loss_ablation/run_ablation.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_${pred_len}_loss_ablation_timesnet" \
        --model "$model_name" \
        --data "$data_name" \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 7 \
        --e_layers 2 \
        --n_heads 8 \
        --d_model $d_model \
        --d_ff $d_ff \
        --des 'Exp' \
        --itr 1 \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs 10 \
        --lradj 'type1' \
        --ma_type $ma_type \
        --alpha $alpha \
        --beta $beta \
        --top_k 5 \
        --num_kernels 6 \
        2>&1 | tee -a "$log_file"
    done
  done
done
