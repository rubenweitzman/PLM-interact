# Model Download Guide

## Do You Need to Download Models?

### Short Answer:
- **PLM-interact checkpoint**: **YES** - Must download manually (or use HuggingFace repo ID)
- **ESM-2 base model**: **NO** - Auto-downloads from HuggingFace

### Detailed Explanation:

#### 1. PLM-interact Checkpoint (`pytorch_model.bin`)

**You MUST download this manually** because:
- It's a custom trained model, not a standard HuggingFace model
- The code uses `torch.load()` which requires a local file path
- It's ~2.6 GB (650M models) or ~140 MB (35M models)

**Two ways to provide it:**

**Option A: Download manually first (recommended)**
```bash
python download_model.py --model humanV12
# Then use:
--checkpoint-path ./models/PLM-interact-650M-humanV12/pytorch_model.bin
```

**Option B: Use HuggingFace repo ID (auto-downloads)**
```bash
# The script will auto-download if checkpoint not found:
--checkpoint-path danliu1226/PLM-interact-650M-humanV12
```

#### 2. ESM-2 Base Model

**Auto-downloads** if you use HuggingFace model ID:
- `facebook/esm2_t33_650M_UR50D` - Auto-downloads ✅
- `facebook/esm2_t12_35M_UR50D` - Auto-downloads ✅

**Or download offline** (optional, for faster loading):
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save offline
offline_path = "../offline/test/esm2_t33_650M_UR50D"
model.save_pretrained(offline_path)
tokenizer.save_pretrained(offline_path)
```

## Recommended Workflow

### Step 1: Download PLM-interact Checkpoint
```bash
cd eval_profluent_style
python download_model.py --model humanV12
```

This downloads to: `./models/PLM-interact-650M-humanV12/pytorch_model.bin`

### Step 2: Run Evaluation
```bash
python eval_profluent_style.py \
    --checkpoint-path ./models/PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    ...
```

**Note**: The ESM-2 model (`esm2_t33_650M_UR50D`) will auto-download from HuggingFace if not found in `../offline/test/`.

## Alternative: Auto-Download Checkpoint

The script now supports auto-downloading the checkpoint if you provide a HuggingFace repo ID:

```bash
python eval_profluent_style.py \
    --checkpoint-path danliu1226/PLM-interact-650M-humanV12 \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    ...
```

This will:
1. Check if checkpoint exists locally
2. If not, download from HuggingFace automatically
3. Save to `./models/PLM-interact-650M-humanV12/`
4. Continue with inference

## Summary Table

| Component | Auto-Download? | Size | Required? |
|-----------|---------------|------|-----------|
| **PLM-interact checkpoint** | ❌ No (unless using repo ID) | 2.6 GB / 140 MB | ✅ Yes |
| **ESM-2 base model** | ✅ Yes (if using HF ID) | ~1.3 GB | ✅ Yes |

## Quick Test

To verify everything works:

```bash
# Download model
python download_model.py --model humanV12

# Test pipeline
python test_pipeline.py \
    --checkpoint-path ./models/PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603
```

The ESM-2 model will auto-download if not found offline!

