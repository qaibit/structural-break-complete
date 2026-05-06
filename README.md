# Structural Break Complete — by CONDOR

> **Sovereign Intelligence** · [condor.qaibit.com](https://condor.qaibit.com)

The full, state-of-the-art structural break detection system combining gradient boosting experts with **PINT-Seq** (Parallel Interactive Neural Transformers for Sequences). This is the **Complete** variant — for a lighter version without neural components, see [structural-break-lite](https://github.com/qaibit/structural-break-lite).

---

## Table of Contents

1. [What is Structural Break Detection?](#what-is-structural-break-detection)
2. [Architecture Overview](#architecture-overview)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Installation (Step by Step)](#installation-step-by-step)
6. [Data Format](#data-format)
7. [Quick Start](#quick-start)
8. [Training & Testing Guide](#training--testing-guide)
9. [Understanding the Output](#understanding-the-output)
10. [Models in Detail](#models-in-detail)
11. [Configuration & Tuning](#configuration--tuning)
12. [Troubleshooting](#troubleshooting)
13. [Citation](#citation)
14. [License](#license)

---

## What is Structural Break Detection?

A **structural break** is a sudden, significant change in the statistical properties of a time series — for example, a shift in mean, variance, or autocorrelation structure. Detecting these breaks is critical in quantitative finance for:

- Identifying regime changes in asset returns
- Detecting anomalies in trading signals
- Risk management and portfolio rebalancing

This system was developed for the [CrunchDAO Structural Break Open Benchmark](https://www.crunchdao.com/), where it achieved **91% AUC** and a **Top 6 global ranking**.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        INPUT DATA                             │
│          DataFrame with MultiIndex (id, time)                 │
│          Columns: value, period                               │
└──────────────────┬───────────────────────────────────────────┘
                   │
     ┌─────────────▼──────────────┐
     │     FEATURE ENGINEERING     │
     │                             │
     │  ┌───────────────────────┐  │
     │  │  Impl2 Features       │  │ Statistical, spectral, distributional
     │  │  (600+ features)      │  │ KS, CUSUM, MMD, Wasserstein, energy
     │  └───────────────────────┘  │
     │  ┌───────────────────────┐  │
     │  │  Impl3 Features       │  │ Quantile-based time-domain features
     │  │  (50+ features)       │  │ Delta features across periods
     │  └───────────────────────┘  │
     │  ┌───────────────────────┐  │
     │  │  Comprehensive        │  │ Robust CVs, IQR, IDR, MAD
     │  │  (80+ features)       │  │ Pre/post period comparisons
     │  └───────────────────────┘  │
     │  ┌───────────────────────┐  │
     │  │  PINT Features        │  │ Physics-Informed Neural features
     │  │  (neural embeddings)  │  │ SHO-prior AR residuals
     │  └───────────────────────┘  │
     └─────────────┬───────────────┘
                   │
     ┌─────────────▼──────────────┐
     │     FEATURE SELECTION       │
     │                             │
     │  1. KS-Shift Filter         │ Removes drifting features
     │  2. Fold-wise MI Selection  │ Top-420 informative features
     └─────────────┬───────────────┘
                   │
     ┌─────────────▼──────────────────────────────────────┐
     │                   BASE MODELS                       │
     │                                                     │
     │  ┌─── Gradient Boosting ───┐  ┌─── Neural ────────┐│
     │  │ • HGB                   │  │ • PINT-Seq v3.0   ││
     │  │ • HGB Distance          │  │   (multi-window)  ││
     │  │ • CatBoost A (Bayesian) │  │ • PINT-Hybrid     ││
     │  │ • CatBoost B (Bernoulli)│  │   (CatBoost+PINT) ││
     │  │ • XGBoost (Impl3)       │  └───────────────────┘│
     │  │ • XGBoost (MI feats)    │                        │
     │  │ • CatBoost Multi-Window │                        │
     │  └─────────────────────────┘                        │
     └─────────────┬───────────────────────────────────────┘
                   │
     ┌─────────────▼──────────────┐
     │     ENSEMBLE BLENDING       │
     │                             │
     │  1. Dirichlet Rank Blend    │ 4500 random + 2500 refine
     │  2. SLSQP Optimization     │ Constrained weights
     │  3. Meta-Stacker (LR)       │ Logistic Regression stacker
     │  4. Alpha Mix               │ Blend RB + Meta-stacker
     │                             │
     │  Output: break_score        │
     └─────────────────────────────┘
```

---

## Repository Structure

```
structural-break-complete/
├── README.md                    ← You are here
├── requirements.txt             ← Python dependencies
├── main.py                      ← Entry point: run_complete_inference()
├── ensemble_expertos_pint.py    ← Orchestrator: combines all modules
├── expertos_8642.py             ← Core engine: features, boosting models,
│                                   blending (2300+ lines)
├── pint_7326.py                 ← PINT-Seq v2.1: windowed sequence models,
│                                   CatBoost multi-window, calibration
└── pint_seq_v3_optimized.py     ← PINT-Seq v3.0: optimized transformer
                                    architecture with attention
```

---

## Requirements

| Package         | Version  | Purpose                                 |
|-----------------|----------|-----------------------------------------|
| Python          | ≥ 3.9    | Runtime                                 |
| pandas          | ≥ 1.5    | Data manipulation                       |
| numpy           | ≥ 1.23   | Numerical computing                     |
| scikit-learn    | ≥ 1.2    | HGB, Logistic Regression, MI            |
| scipy           | ≥ 1.10   | Statistical tests, optimization         |
| catboost        | ≥ 1.2    | CatBoost models                         |
| xgboost         | ≥ 1.7    | XGBoost models                          |
| **torch**       | ≥ 2.0    | **PINT-Seq neural network (required)**  |
| tqdm            | ≥ 4.64   | Progress bars                           |
| statsmodels     | ≥ 0.14   | Ljung-Box test (optional)               |

> ⚠️ **PyTorch is required** for the Complete variant. Without it, the PINT-Seq components will be disabled and you'll get a reduced ensemble (equivalent to Lite).

---

## Installation (Step by Step)

### 1. Clone the repository

```bash
git clone https://github.com/qaibit/structural-break-complete.git
cd structural-break-complete
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your platform. Examples:

```bash
# CPU only (all platforms)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# macOS with Apple Silicon (MPS acceleration)
pip install torch

# Linux with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify installation

```python
python -c "
import torch
from main import run_complete_inference
print(f'✅ PyTorch {torch.__version__} — Device: {\"cuda\" if torch.cuda.is_available() else \"mps\" if hasattr(torch.backends, \"mps\") and torch.backends.mps.is_available() else \"cpu\"}')
print('✅ CONDOR Complete ready')
"
```

---

## Data Format

Your data must follow the **CrunchDAO Structural Break** format:

### X_train / X_test (Features)

A DataFrame with a **MultiIndex** of two levels: `(id, time)` and two columns:

| Column   | Type    | Description                              |
|----------|---------|------------------------------------------|
| `value`  | float64 | The observed time series value           |
| `period` | int     | 0 = pre-break period, 1 = post-break     |

```
                    value  period
id       time                    
series_0 0      0.234521       0
         1      0.198432       0
         ...
         870    0.543210       1
         871    0.567890       1
series_1 0     -0.112345       0
         ...
```

### y_train / y_test (Labels)

A Series or single-column DataFrame indexed by `id`:

| id       | target |
|----------|--------|
| series_0 | 1      |
| series_1 | 0      |
| series_2 | 1      |

Where `1` = structural break, `0` = no break.

### Loading from Parquet

```python
import pandas as pd

X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")
y_test  = pd.read_parquet("data/y_test.parquet")  # optional

print(f"Training: {X_train.index.get_level_values('id').nunique()} series")
print(f"Test:     {X_test.index.get_level_values('id').nunique()} series")
```

---

## Quick Start

```python
from ensemble_expertos_pint import run_all_combined

results = run_all_combined(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test       # pass None if you don't have labels
)

print(f"Ensemble AUC: {results['oof_meta_mix']:.4f}")
print(f"Test AUC:     {results['test_auc']:.4f}")
```

---

## Training & Testing Guide

### Step 1: Prepare your data

Place Parquet files in a `data/` directory:

```
structural-break-complete/
├── data/
│   ├── X_train.parquet
│   ├── y_train.parquet
│   ├── X_test.parquet
│   └── y_test.parquet       # optional
├── main.py
├── ensemble_expertos_pint.py
├── expertos_8642.py
├── pint_7326.py
├── pint_seq_v3_optimized.py
└── ...
```

### Step 2: Run the full pipeline

Create a script `run.py`:

```python
#!/usr/bin/env python3
"""Run the CONDOR Complete structural break detection pipeline."""

import pandas as pd
from ensemble_expertos_pint import run_all_combined

# ── 1. Load data ──
print("Loading data...")
X_train = pd.read_parquet("data/X_train.parquet")
y_train = pd.read_parquet("data/y_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")

try:
    y_test = pd.read_parquet("data/y_test.parquet")
    print(f"✅ Test labels loaded ({len(y_test)} series)")
except FileNotFoundError:
    y_test = None
    print("⚠️  No test labels — inference only")

# ── 2. Run full ensemble ──
results = run_all_combined(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    use_pint=True,           # Enable PINT-Seq neural models
    use_pint_hybrid=True,    # Enable PINT-Hybrid (CatBoost + PINT)
    use_meta_in_mix=True,    # Enable Meta-Stacker LR
)

# ── 3. Print results ──
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\n📊 Per-model OOF AUC:")
for model, auc in sorted(results['oof'].items(), key=lambda x: x[1], reverse=True):
    print(f"   {model:20s}  →  {auc:.4f}")

print(f"\n🎯 Rank-Blend AUC:   {results['oof_blend']:.4f}")
print(f"🧠 Meta-Stacker AUC: {results['oof_meta']:.4f}")
print(f"🏆 Final Mix AUC:    {results['oof_meta_mix']:.4f}")

if results.get('test_auc') is not None:
    print(f"📈 Test AUC:         {results['test_auc']:.4f}")

if results.get('test_auc_no_gcv') is not None:
    print(f"📈 Test AUC (no gcv): {results['test_auc_no_gcv']:.4f}")

print("\n⚖️  Blend Weights:")
for model, weight in sorted(results['weights'].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(weight * 50)
    print(f"   {model:20s}  {weight:.3f}  {bar}")

if results.get('feature_importance'):
    print("\n🔍 Feature Importance (Meta-Stacker):")
    for model, imp in sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {model:20s}  {imp:.6f}")

print(f"\n💾 Predictions saved to: submission_stack_impl2_impl3.csv")
```

Run it:

```bash
python run.py
```

### Step 3: Run WITHOUT neural components

If you don't have PyTorch or want a faster run:

```python
results = run_all_combined(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    use_pint=False,           # Disable PINT-Seq
    use_pint_hybrid=False,    # Disable PINT-Hybrid
)
```

### Step 4: Restrict to specific models

```python
results = run_all_combined(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    allowed_models=["hgb", "cbA", "cbB", "pint_seq_48"]
)
```

### Step 5: Read predictions

```python
import pandas as pd

predictions = pd.read_csv("submission_stack_impl2_impl3.csv", index_col="id")
print(predictions.head())
#              break_score
# id                      
# series_0       0.891234
# series_1       0.045678
# series_2       0.978901
```

Higher `break_score` = higher probability of a structural break.

---

## Understanding the Output

The `run_all_combined()` function returns:

| Key                    | Type              | Description                                       |
|------------------------|-------------------|---------------------------------------------------|
| `oof`                  | dict[str, float]  | Out-of-fold AUC for each base model               |
| `oof_blend`            | float             | Rank-blend ensemble AUC                            |
| `oof_meta`             | float             | Meta-stacker (LR) AUC                             |
| `oof_meta_mix`         | float             | Final blended AUC (rank-blend + meta-stacker)      |
| `weights`              | dict[str, float]  | Optimal blend weight per model                     |
| `feature_importance`   | dict[str, float]  | LR coefficient magnitude per model                 |
| `test_auc`             | float or None     | Test AUC (if `y_test` provided)                    |
| `oof_meta_mix_no_gcv`  | float or None     | AUC of parallel branch (without `global_cv_var`)   |
| `test_auc_no_gcv`      | float or None     | Test AUC of parallel branch                        |

---

## Models in Detail

### Gradient Boosting Models (from `expertos_8642.py`)

| Model       | Key         | Description                                          |
|-------------|-------------|------------------------------------------------------|
| HGB         | `hgb`       | scikit-learn HistGradientBoosting on MI features      |
| HGB Dist    | `dist`      | HGB on distribution-shift features only               |
| CatBoost A  | `cbA`       | Bayesian bootstrap, 3 seeds, 3200 iterations          |
| CatBoost B  | `cbB`       | Bernoulli bootstrap, 2 seeds, 4000 iterations         |
| XGBoost Raw | `xgb_raw`   | XGBoost on Impl3 quantile features                    |
| XGBoost MI  | `xgb_xmi`   | XGBoost on MI-selected features                       |

### Neural Models (from `pint_7326.py` and `pint_seq_v3_optimized.py`)

| Model            | Key              | Description                                  |
|------------------|------------------|----------------------------------------------|
| PINT-Seq         | `pint_seq_*`     | Windowed LSTM with SHO physics prior          |
| PINT-Hybrid      | `pint_hybrid`    | CatBoost on PINT neural embeddings            |
| CatBoost Window  | `cb_*_*`         | CatBoost on windowed sequence features        |
| PINT-Seq v3.0    | `pint_seq_*`     | Optimized transformer with attention          |

### Blending Strategy

1. **Rank Normalization**: All model predictions are converted to percentile ranks (0-1)
2. **Dirichlet Search**: 4500 random weight vectors sampled from Dirichlet distribution
3. **Refinement**: 2500 trials concentrated around the best weights found
4. **SLSQP**: Constrained optimization to fine-tune weights
5. **Meta-Stacker**: Logistic Regression trained on OOF predictions as features
6. **Alpha Mix**: Optimal linear combination of rank-blend and meta-stacker

---

## Configuration & Tuning

### `expertos_8642.py` — Core parameters

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `SEED`                 | 42      | Random seed                                    |
| `FOLDS`                | 5       | Cross-validation folds                         |
| `FAST_MODE`            | False   | Quick mode (less accurate)                     |
| `TOPK_FULL`            | 420     | Features after MI selection                    |
| `BLEND_RANDOM_TRIALS`  | 4500    | Dirichlet search trials                        |

### `pint_7326.py` — PINT-Seq parameters

| Parameter       | Default | Description                           |
|-----------------|---------|---------------------------------------|
| `SEQ_WINDOWS`   | varies  | Window sizes for sequence extraction  |
| `WINDOWS_CB`    | varies  | Window sizes for CatBoost windowed    |

### `pint_seq_v3_optimized.py` — PINT v3 parameters

| Parameter       | Default | Description                           |
|-----------------|---------|---------------------------------------|
| `SEQ_WINDOWS_V3`| varies  | Optimized window sizes                |
| `EPOCHS_SEQ_V3` | varies  | Training epochs for v3                |
| `LR_SEQ_V3`     | varies  | Learning rate for v3                  |

### Enable Fast Mode

```python
# In expertos_8642.py, change:
FAST_MODE = True
```

---

## Troubleshooting

### "PyTorch no disponible"

Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/):
```bash
pip install torch
```

### "CatBoost no disponible"
```bash
pip install catboost
```

### "XGBoost no disponible"
```bash
pip install xgboost
```

### Out of memory

1. Enable `FAST_MODE = True`
2. Set `use_pint=False` to disable neural models
3. Reduce `TOPK_FULL` from 420 to 200
4. Use `allowed_models` to restrict the ensemble

### CUDA errors

If you get GPU memory errors, force CPU:
```python
# In expertos_8642.py, change:
PINT_DEVICE = "cpu"
```

### Slow execution

Expected runtimes (on Apple M2, 8GB RAM):
- **Lite** (no PINT): ~5-15 minutes
- **Complete** (with PINT): ~30-60 minutes
- **Complete + CUDA**: ~10-20 minutes

---

## Citation

If you use this code in your research or projects, please cite:

```
@software{condor_structural_break_complete,
  author = {CONDOR},
  title = {Structural Break Detection (Complete) — Sovereign Intelligence},
  year = {2026},
  publisher = {Qaibit},
  url = {https://github.com/qaibit/structural-break-complete}
}
```

---

## Authors

Developed by **CONDOR** — Sovereign Intelligence.

- 🌐 Platform: [condor.qaibit.com](https://condor.qaibit.com)
- 🏢 Organization: [Qaibit](https://qaibit.com)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
