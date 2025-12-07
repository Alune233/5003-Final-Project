# LLM-Generated Content Detection with HPO Methods Comparison

**DSAA 5003 Final Project - AutoML and Hyperparameter Optimization**

## Abstract

This project applies Automated Machine Learning (AutoML) techniques, specifically focusing on Hyperparameter Optimization (HPO), to solve the LLM-generated content detection problem. We systematically compare four different HPO methods (Random Search, Grid Search, TPE via Optuna, and SMAC via OpenBox) across three diverse machine learning models (LightGBM, SVM, and MLP). Our experimental framework emphasizes reproducibility, comprehensive analysis, and fair comparison through controlled variables and unified search spaces.

## Team Structure

| Member               | Role            | Code Responsibilities                                                | Report Sections                    |
| -------------------- | --------------- | -------------------------------------------------------------------- | ---------------------------------- |
| **Jiawei He**  | Data & Baseline | Feature extraction (DeBERTa), search space definition, Random Search | Introduction, Data Preprocessing   |
| **Ling Zhao**  | Grid Search     | Grid Search implementation, model analysis                           | Related Work, Model Description    |
| **Ran Mei**    | TPE/Optuna      | TPE optimization implementation                                      | HPO Algorithms, Experimental Setup |
| **Bowen Xiao** | SMAC/OpenBox    | SMAC implementation, result analysis                                 | Experiments & Analysis, Conclusion |

## Project Overview

### Problem Statement

LLM-generated content detection aims to identify which Large Language Model generated a given text based on question-answer pairs. This is a multi-class classification problem with 7 target categories.

### Key Features

- **Controlled Experiment Design**: Unified DeBERTa embeddings and consistent search spaces across all methods
- **Comprehensive HPO Comparison**: Four different optimization strategies evaluated systematically
- **Multi-Model Analysis**: Three diverse models representing different algorithm paradigms
- **Reproducible Pipeline**: Automated workflow with minimal configuration required

## Repository Structure

```
.
├── main.py                          # Main entry point for experiments
├── config/
│   └── search_spaces.json           # Unified hyperparameter search spaces
├── src/
│   ├── models/                      # Model implementations
│   │   ├── lgb_model.py            # LightGBM (Gradient Boosting)
│   │   ├── svm_model.py            # Support Vector Machine
│   │   └── mlp_model.py            # Multi-Layer Perceptron
│   ├── hpo/                        # HPO algorithm implementations
│   │   ├── random_search.py        # Random Search (Baseline)
│   │   ├── grid_search.py          # Grid Search
│   │   ├── tpe_optuna.py           # TPE via Optuna
│   │   └── smac_optimizer.py       # SMAC via OpenBox
│   ├── feature_extraction.py       # DeBERTa feature extraction
│   ├── preprocess_features.py      # Feature standardization
│   └── extract_n_trials.py         # Tool for extracting N-trial results
├── data/
│   ├── raw/                        # Original Kaggle data
│   └── processed/                  # Extracted features (.npy)
├── models/                         # Saved models and optimization history
├── outputs/                        # Kaggle submission files
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for LightGBM acceleration)

### Dependencies

```bash
# Core dependencies
pip install numpy pandas scikit-learn lightgbm torch transformers matplotlib tqdm

# HPO libraries
pip install optuna      # For TPE (Member C)
pip install openbox     # For SMAC (Member D)
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Feature Extraction (Run Once)

Extract DeBERTa embeddings from raw text data:

```bash
python main.py --mode extract
```

This generates feature files in `data/processed/`:

- `train_features.npy` - Training features (768-dim DeBERTa embeddings)
- `test_features.npy` - Test features
- `train_labels.npy` - Training labels
- `test_ids.npy` - Test IDs

### Step 2: Feature Standardization (Run Once)

Standardize features for SVM and MLP convergence:

```bash
python src/preprocess_features.py
```

This creates standardized features and backs up original files.

### Step 3: Run HPO Experiments

**General Command Format:**

```bash
python main.py --model [MODEL] --algo [ALGORITHM] --n_trials [N]
```

**Available Options:**

- Models: `lightgbm`, `svm`, `mlp`
- Algorithms: `random`, `grid`, `tpe`, `smac`
- n_trials: Number of trials (default: 50)

**Example Experiments:**

```bash
# Random Search (Baseline)
python main.py --model lightgbm --algo random --n_trials 50
python main.py --model svm --algo random --n_trials 50
python main.py --model mlp --algo random --n_trials 50

# Grid Search
python main.py --model lightgbm --algo grid
python main.py --model svm --algo grid
python main.py --model mlp --algo grid

# TPE (Optuna)
python main.py --model lightgbm --algo tpe --n_trials 50
python main.py --model svm --algo tpe --n_trials 50
python main.py --model mlp --algo tpe --n_trials 50

# SMAC (OpenBox)
python main.py --model lightgbm --algo smac --n_trials 50
python main.py --model svm --algo smac --n_trials 50
python main.py --model mlp --algo smac --n_trials 50
```

### Step 4: Extract N-Trial Results (Optional)

To compare performance at different trial counts (10, 20, 50):

```bash
# Extract first 10 trials from all experiments
python src/extract_n_trials.py --n_trials 10

# Extract first 20 trials
python src/extract_n_trials.py --n_trials 20
```

This automatically processes all history files in `models/` and generates:

- `{model}_{algo}_{N}trials_history.json` - Optimization history
- `{model}_{algo}_{N}trials_history.png` - Convergence plot
- `{model}_{algo}_{N}trials_submission.csv` - Kaggle submission

## Experimental Design

### Controlled Variables

1. **Data**: Unified DeBERTa embeddings (768 dimensions) for all experiments
2. **Models**: Three models representing different paradigms:
   - LightGBM: Gradient Boosting Decision Trees
   - SVM: Kernel-based large-margin classifier
   - MLP: Neural network with adaptive learning
3. **Search Space**: Consistent parameter ranges defined in `config/search_spaces.json`
4. **Evaluation**: 5-fold stratified cross-validation with log-loss metric

### Independent Variable

- **HPO Method**: Random Search, Grid Search, TPE, SMAC

### Evaluation Metrics

- **Primary**: Log-loss (lower is better)
- **Secondary**: Convergence speed, computational efficiency
- **Analysis**: Performance vs. trial count, parameter importance

## Output Files

Each experiment generates:

```
models/
├── {model}_{algo}_history.json      # Optimization history for analysis
├── {model}_{algo}_history.png       # Convergence visualization
└── {model}_fold_*.pkl               # Trained model weights

outputs/
└── {model}_{algo}_submission.csv    # Kaggle submission file
```

## Results Summary

Our experiments systematically compare 4 HPO methods × 3 models = 12 configurations. Key findings include:

1. **Method Comparison**: TPE and SMAC demonstrate superior convergence on complex search spaces (MLP)
2. **Model Performance**: LightGBM achieves best overall performance with proper tuning
3. **Efficiency Analysis**: Grid Search exhaustive but limited to discrete spaces; Random Search provides strong baseline

_(Detailed results and analysis are provided in the report)_

## Reproducibility

### Search Space Configuration

Hyperparameter search spaces are defined in `config/search_spaces.json` and shared across all HPO methods (except Grid Search, which uses discretized grids in `src/hpo/grid_search.py`).

### Random Seeds

- Fixed random seed: 42
- Deterministic cross-validation splits
- Reproducible model training

### Hardware

- CPU: Multi-core processors (utilized by all methods)
- GPU: Optional, auto-detected for LightGBM acceleration
- Memory: ~8GB RAM recommended

## External Resources & Citations

### Datasets

- **Kaggle Competition**: [LLM Generated Content Detection Challenge]
- Original dataset provided by competition organizers

### Pre-trained Models

- **DeBERTa-v3-base**: Microsoft's DeBERTa model via HuggingFace Transformers
  - Citation: He, P., et al. (2021). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
    ### Code References
