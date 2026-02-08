#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Create results directory
mkdir -p loss_ablation/results
log_file="loss_ablation/results/loss_ablation_smamba_tuned.log"

# Clear previous log
echo "Starting smamba Tuned Experiment (Arctangent Loss, alpha=0.3)" > "$log_file"

# Models to compare
# 1. smamba (Tuned)
models=("smamba")

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
      
      # Determine hyperparameters based on dataset and pred_len
      # Default values
      d_model=512 # Fallback if not set
      d_ff=2048   # Fallback if not set
      learning_rate=0.0001 # Fallback if not set
      
      if [ "$data_name" == "ETTh1" ]; then
        d_model=256
        d_ff=256
        if [ "$pred_len" == "96" ] || [ "$pred_len" == "192" ]; then
            learning_rate=0.00007
        else
            learning_rate=0.00005
        fi
      elif [ "$data_name" == "ETTh2" ]; then
        d_model=256
        d_ff=256
        if [ "$pred_len" == "96" ] || [ "$pred_len" == "192" ]; then
            learning_rate=0.00004
        elif [ "$pred_len" == "336" ]; then
            learning_rate=0.00003
        else
            learning_rate=0.00007
        fi
      elif [ "$data_name" == "ETTm1" ]; then
        if [ "$pred_len" == "96" ]; then
            d_model=256
            d_ff=256
        else
            d_model=128
            d_ff=128
        fi
        learning_rate=0.00005
      elif [ "$data_name" == "ETTm2" ]; then
        if [ "$pred_len" == "96" ]; then
            d_model=256
            d_ff=256
        else
            d_model=128
            d_ff=128
        fi
        
        if [ "$pred_len" == "336" ]; then
            learning_rate=0.00003
        else
            learning_rate=0.00005
        fi
      fi

      python -u loss_ablation/run_ablation.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_${pred_len}_loss_ablation_tuned" \
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
        --d_state 2 \
        --des 'Exp' \
        --itr 1 \
        --batch_size 32 \
        --learning_rate $learning_rate \
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
