#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
seq_len=96
pred_len=96 

# Create results directory if it doesn't exist
mkdir -p results/ablation

# Log file
log_file="results/ablation/ablation_alpha_dmamba_ett.log"
echo "Starting Alpha Ablation for DMamba (EMA) on ETT datasets" > "$log_file"

# Alphas to test (excluding 0.3)
alphas=(0.1 0.5 0.7 0.9)

# Datasets
datasets=("ETTh1.csv" "ETTh2.csv" "ETTm1.csv" "ETTm2.csv")

for alpha in "${alphas[@]}"
do
  for data_path in "${datasets[@]}"
  do
    data_name=${data_path%.csv}
    
    echo "=========================================================" | tee -a "$log_file"
    echo "Running $data_name with alpha=$alpha" | tee -a "$log_file"
    echo "=========================================================" | tee -a "$log_file"
    
    # Using run.py which uses Exp_Main (Arctan Loss by default for DMamba)
    # Removed --n_heads as it is not a valid argument for run.py
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path "$data_path" \
      --model_id "${data_name}_${pred_len}_alpha${alpha}" \
      --model DMamba \
      --data "$data_name" \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 2 \
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
      --beta 0.3 \
      2>&1 | tee -a "$log_file"
      
  done
done
