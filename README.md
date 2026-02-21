# Temporal-Graph-Neural-Network-TGNN-for-Causal-Asset-Relationships

## Overview
Financial markets are highly interconnected ecosystems where an event in one asset class quickly ripples into others. This repository contains a deep learning pipeline that uses a Temporal Graph Neural Network (TGNN) to model dynamic causal relationships across various financial assets. 

Instead of relying on static correlation matrices, this model learns time-varying adjacency matrices using multi-head attention over edges—where the attention weight approximates the influence strength between assets.

## Asset Universe
The system downloads real market data via `yfinance` and evaluates 25 distinct assets:
* **11 Equity Sector ETFs** (e.g., XLK, XLF, XLE)
* **6 Commodity ETFs** (e.g., GLD, USO, DBA)
* **8 FX Rates** (e.g., EUR/USD via FXE, JPY/USD via FXY)

## Key Features & Architecture
1. **Feature Engineering:** Extracts critical technical indicators including 1-day/5-day/21-day returns, rolling volatilities, momentum, RSI-14, and rolling 63-day z-scores.
2. **Temporal Encoder:** Utilizes a Gated Recurrent Unit (GRU) to summarize the historical lookback window (temporal dynamics) independently per asset.
3. **Graph Attention Layer:** Computes a directed, sparse attention-weighted adjacency matrix to learn "how much asset $j$ explains asset $i$". 
4. **Sparsity Regularization:** Applies L1 sparsity constraints during training to filter out spurious edges and maintain a robust causal graph.
5. **Graph Convolutions:** Passes messages over the learned graph to predict next-day, 1-step-ahead returns for each asset.

## Notebook Structure
The core workflow is contained entirely within TGNN Causal asset relationships.ipynb:

**Section 1 & 2** - Setup and Real Data Download: Initializes dependencies, configurations, and downloads data directly from Yahoo Finance from 2018 to 2024.

**Section 3 & 4** - Feature Engineering & Dataset Generation: Computes rolling features and packages them into batched sequences for the PyTorch DataLoader.

**Section 5** - Model Architecture: Defines the Temporal Encoder, Graph Attention Layer, Graph Convolution Layers, and the final combined TemporalCausalGNN.

**Section 6** - Training: Executes the training loop with early stopping, tracking MSE losses and sparsity penalties.

**Section 7** - Evaluation & Visualization: Evaluates Graph Statistics (average edge density/stability), model metrics (Directional Accuracy, Information Coefficient), and overlaps with traditional Granger Causality tests.

## Disclaimer
This repository is for research and educational purposes only. The models and signals generated do not constitute financial advice.

## Getting Started

### Prerequisites
Ensure you have Python 3 installed and a machine compatible with PyTorch (CPU or CUDA). 

```bash
git clone [https://github.com/yourusername/tgnn-causal-assets.git](https://github.com/yourusername/tgnn-causal-assets.git)
cd tgnn-causal-assets
pip install -r requirements.txt
