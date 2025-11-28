# Quick Start Guide: Download Model & Test Pipeline

## Option 1: Quick Test (All-in-One)

Run the automated test script that downloads the model and tests the pipeline:

```bash
cd eval_profluent_style
bash quick_test.sh
```

Or specify a different model:
```bash
bash quick_test.sh 35M  # For smaller, faster model
```

## Option 2: Step-by-Step

### Step 1: Download Model

Download the recommended model (PLM-interact-650M-humanV12):

```bash
python download_model.py --model humanV12
```

This will download the model to `./models/PLM-interact-650M-humanV12/`

**Other models available:**
- `humanV11` - Alternative human PPI model
- `35M` - Smaller, faster model
- `VH` - Virus-human interactions
- `leakage-free` - Leakage-free evaluation
- `mutation` - Mutation effect prediction

**Example:**
```bash
# Download smaller model
python download_model.py --model 35M --output-dir ./models/PLM-interact-35M

# Force re-download
python download_model.py --model humanV12 --force
```

### Step 2: Test Pipeline

Test the pipeline with a small example dataset:

```bash
python test_pipeline.py \
    --checkpoint-path ./models/PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --num-samples 5 \
    --batch-size 2
```

**For 35M model:**
```bash
python test_pipeline.py \
    --checkpoint-path ./models/PLM-interact-35M-humanV11/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t12_35M_UR50D \
    --embedding-size 480 \
    --max-length 1603 \
    --num-samples 5 \
    --batch-size 4
```

### Step 3: Run Full Evaluation

Once the test passes, run on a real dataset:

```bash
python eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ./models/PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi
```

## Prerequisites

1. **Install dependencies:**
```bash
pip install huggingface_hub click pandas numpy torch transformers
```

2. **ESM-2 base model:**
   - The ESM-2 model will be automatically downloaded by HuggingFace Transformers
   - Or download offline: `python -c "from transformers import AutoModelForMaskedLM; AutoModelForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D').save_pretrained('../offline/test/esm2_t33_650M_UR50D')"`

3. **GPU (recommended):**
   - CUDA-enabled GPU with at least 8GB VRAM for 650M models
   - Or use CPU (slower) or 35M model for less memory

## Troubleshooting

### Model Download Issues

**Error: "huggingface_hub not found"**
```bash
pip install huggingface_hub
```

**Error: "Not enough disk space"**
- 650M models need ~3 GB
- 35M models need ~200 MB
- Check available space: `df -h`

**Error: "Connection timeout"**
- Check internet connection
- Try again (downloads can be slow)
- Use `--force` to resume interrupted downloads

### Pipeline Test Issues

**Error: "Checkpoint not found"**
- Make sure you downloaded the model first
- Check the path is correct
- Use absolute path if relative path doesn't work

**Error: "CUDA out of memory"**
- Reduce batch size: `--batch-size 1`
- Use 35M model instead: `--model 35M`
- Use CPU: `--device cpu` (in test_pipeline.py)

**Error: "Offline model path not found"**
- The ESM-2 model will auto-download if not found offline
- Or download manually (see Prerequisites)

## Expected Output

### Download Script Output:
```
============================================================
Downloading PLM-interact Model: humanV12
============================================================
Repository: danliu1226/PLM-interact-650M-humanV12
Description: Recommended: Trained on STRING V12 (most recent)
Output directory: ./models/PLM-interact-650M-humanV12
Base model: esm2_t33_650M_UR50D
Embedding size: 1280
Max length: 1603
============================================================

Downloading from HuggingFace...
This may take several minutes (model size: ~2.6 GB for 650M, ~140 MB for 35M)...

============================================================
✓ Download Complete!
============================================================
Model saved to: ./models/PLM-interact-650M-humanV12
Checkpoint file: ./models/PLM-interact-650M-humanV12/pytorch_model.bin
```

### Test Script Output:
```
============================================================
PLM-interact Pipeline Test
============================================================
Checkpoint: ./models/PLM-interact-650M-humanV12/pytorch_model.bin
Model: esm2_t33_650M_UR50D
Embedding size: 1280
Max length: 1603
Test samples: 5
============================================================

[Step 1/4] Creating test dataset...
Created test dataset with 5 samples
Saved to: /tmp/...

[Step 2/4] Extracting protein pairs...
✓ Extracted 5 pairs

[Step 3/4] Creating PLM-interact CSV format...
✓ Created CSV file: /tmp/.../protein_pairs.csv

[Step 4/4] Running PLM-interact inference...
✓ Test Successful!
Processed 5 protein pairs
Prediction range: 0.1234 - 0.9876
Prediction mean: 0.5432

Sample predictions:
  1. MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSMLLNDGILM...
     Prediction: 0.5432
```

## Next Steps

After successful test:
1. Run on your datasets using `eval_profluent_style.py`
2. Check results in `./results/{dataset_name}/results.csv`
3. Compare with other methods (MINT, ProteomeLM)

For more details, see:
- `MODEL_SELECTION.md` - Choosing the right model
- `README.md` - Detailed documentation
- `USAGE_SUMMARY.md` - Usage examples

