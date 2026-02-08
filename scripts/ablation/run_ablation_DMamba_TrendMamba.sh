#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ma_type=ema
alpha=0.3
beta=0.3
seq_len=96

# Define models to run
# 1. DMamba_TrendMamba (Trend uses TMamba, Seasonal uses Patch+CNN)
# 2. DMamba_TMamba (Trend uses MLP, Seasonal uses TMamba)
# 3. DMamba_DualMamba (Trend uses TMamba, Seasonal uses TMamba)
models=("DMamba_TrendMamba" "DMamba_TMamba" "DMamba_DualMamba")
datasets=("ETTm1" "ETTm2" "ETTh1" "ETTh2")

for model_name in "${models[@]}"
do
    for data_name in "${datasets[@]}"
    do
        for pred_len in 96 192 336 720
        do
          echo "========================================================="
          echo "Starting $data_name experiment ($model_name): pred_len=$pred_len"
          echo "========================================================="
          python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path ${data_name}.csv \
            --model_id ${data_name}_${model_name}_$pred_len \
            --model $model_name \
            --data $data_name \
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
            --train_epochs 30
        done
    done
done
