# PLM-interact PPI Inference - Quick Start Guide

## Overview

PLM-interact is a protein-protein interaction (PPI) prediction method that extends protein language models (PLMs) to jointly encode protein pairs and predict their interactions. This guide explains how to use PLM-interact for PPI inference on MDS format datasets.

## How PLM-interact Works

1. **Joint Encoding**: Unlike methods that process proteins independently, PLM-interact takes both protein sequences together as input
2. **ESM-2 Base Model**: Uses ESM-2 (Evolutionary Scale Modeling) as the foundation
3. **Binary Classifier**: Applies a linear classifier on the CLS token embedding to output interaction probability (0-1)

## Key Differences from Other Methods

| Method | Approach | Input Format |
|--------|----------|--------------|
| **PLM-interact** | Joint encoding of pairs | Both sequences together |
| **MINT** | Separate chain embeddings | Concatenated sequences |
| **ProteomeLM** | Independent processing | Individual proteins |

## Required Files

### 1. PLM-interact Checkpoint
Download from HuggingFace:
- `danliu1226/PLM-interact-650M-humanV12` (recommended)
- `danliu1226/PLM-interact-650M-humanV11`
- `danliu1226/PLM-interact-35M-humanV11` (smaller, faster)

The checkpoint file you need is: `pytorch_model.bin` (2.61 GB for 650M models)

### 2. ESM-2 Base Model
The base ESM-2 model will be downloaded automatically by HuggingFace Transformers, or you can download it offline:
- For 650M models: `facebook/esm2_t33_650M_UR50D`
- For 35M models: `facebook/esm2_t12_35M_UR50D`

## Input Format

PLM-interact expects CSV files with these exact column names:
- `query`: First protein sequence
- `text`: Second protein sequence

**Note**: This is different from MINT which uses `Protein_Sequence_1`/`Protein_Sequence_2`.

## Running Inference

### Basic Command:
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

### Parameter Explanation:

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `--checkpoint-path` | Path to `pytorch_model.bin` | `../PLM-interact-650M-humanV12/pytorch_model.bin` |
| `--offline-model-path` | Directory containing ESM-2 model | `../offline/test/` |
| `--model-name` | ESM-2 model name | `esm2_t33_650M_UR50D` or `esm2_t12_35M_UR50D` |
| `--embedding-size` | Embedding dimension | `1280` (650M) or `480` (35M) |
| `--max-length` | Max sequence length | `1603` (default for 650M) |
| `--batch-size` | Batch size | `16` (default) |
| `--device` | GPU device | `cuda:0` (default) |

### Model-Specific Parameters:

**For 650M models** (PLM-interact-650M-*):
- `--model-name esm2_t33_650M_UR50D`
- `--embedding-size 1280`
- `--max-length 1603`

**For 35M models** (PLM-interact-35M-*):
- `--model-name esm2_t12_35M_UR50D`
- `--embedding-size 480`
- `--max-length 1603` (or adjust based on your data)

## Output Format

The script generates:
1. **`results.csv`**: Main results with columns:
   - `data_source`: Original data source
   - `sequence`: Original comma-separated pair
   - `value`: Ground truth value
   - `prediction`: Interaction probability (0-1)

2. **`ppi_results.pkl`**: Detailed results for analysis

## Example Workflow

```bash
# 1. Download model checkpoint (if not already done)
# See README.md for download instructions

# 2. Run inference on a dataset
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ./PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ./offline/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi

# 3. Check results
cat ./results/alignment_skempi/results.csv | head
```

## Troubleshooting

### Issue: "Model mismatch"
**Solution**: Ensure `--model-name` matches the checkpoint's base model. Check the checkpoint's HuggingFace page.

### Issue: "CUDA out of memory"
**Solution**: Reduce `--batch-size`:
```bash
--batch-size 8  # or even 4
```

### Issue: "Max length too short"
**Solution**: Increase `--max-length`:
```bash
--max-length 2000  # or higher based on your longest protein pair
```

### Issue: "Offline model path not found"
**Solution**: Ensure the path ends with `/` and contains the ESM-2 model files:
```bash
--offline-model-path ../offline/test/  # Note the trailing /
```

## Comparison with Other Methods

The script follows the same pattern as MINT and ProteomeLM evaluation scripts:

1. **Load MDS dataset** from GCS
2. **Extract protein pairs** (split by comma)
3. **Create CSV** in method-specific format
4. **Run inference** using method-specific pipeline
5. **Generate output** CSV with predictions

The main difference is the CSV format:
- **MINT**: `Protein_Sequence_1`, `Protein_Sequence_2`
- **ProteomeLM**: FASTA file with unique proteins
- **PLM-interact**: `query`, `text`

## Additional Resources

- Full documentation: `eval_profluent_style/README.md`
- PLM-interact paper: https://www.biorxiv.org/content/10.1101/2024.11.05.622169v2
- HuggingFace models: https://huggingface.co/danliu1226

