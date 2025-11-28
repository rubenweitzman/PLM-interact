# PLM-interact Model Selection Guide

## Quick Reference

| Model | Use Case | Base Model | Embedding Size | Max Length | Size |
|-------|----------|------------|----------------|------------|------|
| **PLM-interact-650M-humanV12** | ✅ **General human PPI (RECOMMENDED)** | `esm2_t33_650M_UR50D` | 1280 | 1603 | 2.61 GB |
| PLM-interact-650M-humanV11 | General human PPI (alternative) | `esm2_t33_650M_UR50D` | 1280 | 1603 | 2.61 GB |
| PLM-interact-35M-humanV11 | Fast inference, limited GPU | `esm2_t12_35M_UR50D` | 480 | 1603 | ~140 MB |
| PLM-interact-650M-VH | Virus-human interactions | `esm2_t33_650M_UR50D` | 1280 | 1603 | 2.61 GB |
| PLM-interact-650M-Leakage-Free | Leakage-free evaluation | `esm2_t33_650M_UR50D` | 1280 | 1603 | 2.61 GB |
| PLM-interact-650M-Mutation | Mutation effect prediction | `esm2_t33_650M_UR50D` | 1280 | 1603 | 2.61 GB |

## Recommended Model: PLM-interact-650M-humanV12

**For general PPI prediction on your MDS datasets, use:**

```bash
--checkpoint-path ../PLM-interact-650M-humanV12/pytorch_model.bin
--model-name esm2_t33_650M_UR50D
--embedding-size 1280
--max-length 1603
```

**Why this model?**
- Most recent training data (STRING V12)
- Best performance on general human PPI tasks
- Well-tested and documented

## Model-Specific Commands

### 1. PLM-interact-650M-humanV12 (Recommended)
```bash
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ../PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi
```

### 2. PLM-interact-650M-humanV11 (Alternative)
```bash
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ../PLM-interact-650M-humanV11/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi
```

### 3. PLM-interact-35M-humanV11 (Faster, Smaller)
```bash
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ../PLM-interact-35M-humanV11/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t12_35M_UR50D \
    --embedding-size 480 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi \
    --batch-size 32  # Can use larger batch size with smaller model
```

### 4. PLM-interact-650M-VH (Virus-Human)
```bash
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ../PLM-interact-650M-VH/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi
```

## Downloading Models

### From HuggingFace:
```python
from huggingface_hub import snapshot_download
import os

# Recommended model
repo_id = "danliu1226/PLM-interact-650M-humanV12"
local_dir = "../PLM-interact-650M-humanV12"

os.makedirs(local_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    force_download=True
)
```

### Available Models on HuggingFace:
- `danliu1226/PLM-interact-650M-humanV12` ⭐ **Recommended**
- `danliu1226/PLM-interact-650M-humanV11`
- `danliu1226/PLM-interact-35M-humanV11`
- `danliu1226/PLM-interact-650M-VH`
- `danliu1226/PLM-interact-650M-Leakage-Free-Dataset`
- `danliu1226/PLM-interact-650M-Mutation`

## Model Training Data

| Model | Training Dataset | Description |
|-------|------------------|-------------|
| humanV12 | STRING V12 | Latest STRING database (2023) |
| humanV11 | Cross-species dataset | D-SCRIPT cross-species benchmark |
| VH | Virus-human dataset | LSTM-PHV virus-human interactions |
| Leakage-Free | Bernett dataset | Leakage-free evaluation dataset |
| Mutation | IntAct mutations | Mutation effect dataset |

## Choosing the Right Model

### Use **PLM-interact-650M-humanV12** if:
- ✅ You're doing general human PPI prediction
- ✅ You want the best performance
- ✅ You have GPU memory (needs ~8GB+ VRAM)

### Use **PLM-interact-35M-humanV11** if:
- ✅ You need faster inference
- ✅ You have limited GPU memory (< 4GB VRAM)
- ✅ You can accept slightly lower accuracy

### Use **PLM-interact-650M-VH** if:
- ✅ You're specifically predicting virus-human interactions
- ✅ Your dataset contains virus proteins

### Use **PLM-interact-650M-Leakage-Free** if:
- ✅ You need to avoid data leakage issues
- ✅ You're doing rigorous evaluation

### Use **PLM-interact-650M-Mutation** if:
- ✅ You're predicting mutation effects on PPIs
- ✅ Your dataset contains mutated proteins

## Parameter Matching

**Important**: The `--model-name`, `--embedding-size`, and `--max-length` parameters MUST match your checkpoint:

### For 650M models (humanV11, humanV12, VH, etc.):
```bash
--model-name esm2_t33_650M_UR50D
--embedding-size 1280
--max-length 1603
```

### For 35M models (humanV11):
```bash
--model-name esm2_t12_35M_UR50D
--embedding-size 480
--max-length 1603
```

## Performance Comparison

Based on the paper and documentation:
- **650M models**: Higher accuracy, slower inference
- **35M models**: Lower accuracy, faster inference (~3-4x faster)

For most use cases, the 650M models provide the best balance of accuracy and performance.

