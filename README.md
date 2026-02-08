# DMamba: Decomposed Mamba for Time Series Forecasting

DMamba is a state-of-the-art time series forecasting model that combines **Series Decomposition** with the **Mamba (State Space Model)** architecture. It effectively separates time series into trend and seasonal components, processing them with specialized streams to capture both linear long-term patterns and non-linear complex dependencies.

## 🌟 Key Features

- **Series Decomposition**: Utilizes moving average (EMA/DEMA) to decompose series into Trend and Seasonal components.
- **Mamba-based Seasonal Stream**: Leverages the Mamba architecture (SSM) to model complex non-linear dependencies in the seasonal component.
- **MLP-based Trend Stream**: Uses a multi-layer linear structure to capture long-term trend patterns efficiently.
- **Channel-Independent Processing**: Standardizes processing across different variates to improve generalization.
- **Extensive Ablation Studies**: Includes various model variants (DMamba_T, DMamba_TrendMamba, DMamba_TMamba, etc.) for detailed analysis.

## 🏗️ Model Architecture

The core `DMamba` model consists of:
1. **RevIN**: Reversible Instance Normalization to handle distribution shift.
2. **Decomposition**: Splitting input into Trend and Seasonality.
3. **Dual Streams**:
   - **Seasonal Stream**: Data Embedding (Inverted) + Mamba Encoder + Projector.
   - **Trend Stream**: Multi-layer MLP with Average Pooling and Layer Normalization.
4. **Fusion**: Linear concatenation of both streams to produce the final prediction.

## 🚀 Getting Started

### 1. Environment Setup

It is recommended to use the `mamba` environment provided in this repository.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets (e.g., `ETTh2.csv`) in the `./dataset/` directory.

### 3. Training & Evaluation

You can reproduce the experiments using the provided shell scripts in the `scripts/` directory.

#### Reproduce ETTh2 Seasonal Experiment:
```bash
bash scripts/etth2/reproduce_etth2_DMamba_seasonal.sh
```

#### Other Ablation Experiments:
```bash
# Example: Run AllMamba ablation
bash scripts/ablation/run_ablation_DMamba_AllMamba.sh
```

## 📂 Project Structure

- `models/`: Core model implementations (DMamba, DMamba_T, DMamba_MLP, DMamba_TMamba, etc.)
- `layers/`: Component layers (Mamba Encoder, Decomposition, Embedding, RevIN)
- `scripts/`: Shell scripts for reproducing experiments on various datasets (ETT, etc.)
- `exp/`: Experiment management logic (`exp_main.py`)
- `run.py`: Entry point for training and testing.
- `requirements.txt`: List of required Python packages.

## 📊 Results

The model outputs results (MSE/MAE) to the `results/` directory and logs to `logs/`.

---
*Note: This project is part of ongoing research into Mamba-based time series forecasting.*
