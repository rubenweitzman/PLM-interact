# Explanation of `eval_profluent_style.py` for PLM-interact

## Overview
This script evaluates PLM-interact (Protein Language Model for Interactions) on protein-protein interaction (PPI) prediction tasks. It loads datasets from Google Cloud Storage (GCS) in MDS (Mosaic Data Shard) format and generates binary PPI predictions.

## What is PLM-interact?

PLM-interact is a method that extends protein language models (PLMs) to predict protein-protein interactions. Unlike methods that use pre-trained PLM features independently, PLM-interact jointly encodes protein pairs to learn their relationships, similar to next-sentence prediction in natural language processing.

**Key Features:**
- Uses ESM-2 (Evolutionary Scale Modeling) as the base model
- Jointly encodes protein pairs (both sequences together)
- Outputs interaction probability scores (0-1 range)
- Trained specifically for PPI prediction tasks

## What the Script Does (Step-by-Step)

### Step 1: Load MDS Dataset (`load_mds_dataset`)
- **Purpose**: Downloads and loads protein interaction data from GCS
- **Input**: GCS path to MDS dataset (e.g., `gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi`)
- **Process**:
  - Creates a temporary local cache directory
  - Uses `StreamingDataset` to stream data from GCS
  - Extracts samples with fields: `sequence`, `value`, `data_source`
  - `sequence`: Comma-separated pair of protein sequences (e.g., "SEQ1,SEQ2")
  - `value`: Ground truth label/value (e.g., binding affinity, interaction score)
  - `data_source`: Source identifier for the data
- **Output**: List of sample dictionaries

### Step 2: Extract Protein Pairs (`extract_protein_pairs`)
- **Purpose**: Parses comma-separated sequences into separate protein pairs
- **Process**:
  - Splits each `sequence` field by comma (first comma only)
  - Creates pairs with `query` (protein1) and `text` (protein2) columns
  - Handles edge cases (empty sequences, missing commas)
- **Output**: List of dictionaries with two protein sequences in PLM-interact format

### Step 3: Create CSV File (`create_csv_file`)
- **Purpose**: Converts pairs into CSV format for PLM-interact processing
- **Output**: CSV file with columns: `query`, `text`
- **Note**: PLM-interact expects these exact column names (not `Protein_Sequence_1`/`Protein_Sequence_2`)

### Step 4: Run PPI Prediction (`run_ppi_prediction`)
This is the core prediction pipeline:

#### 4a. Load PLM-interact Model
- Loads ESM-2 base model from offline path (e.g., `facebook/esm2_t33_650M_UR50D`)
- Initializes `CrossEncoder` with PLM-interact checkpoint (`pytorch_model.bin`)
- Sets model to evaluation mode

#### 4b. Tokenize and Predict
- **Process**:
  - Tokenizes protein pairs using ESM-2 tokenizer
  - Concatenates both sequences with special tokens
  - Passes through ESM-2 base model to get embeddings
  - Extracts CLS token embedding (first token)
  - Applies ReLU activation and linear classifier
  - Outputs sigmoid probability (interaction score)
- **Output**: Probability scores for each protein pair (0-1 range)

### Step 5: Create Output CSV
- Combines original MDS data with predictions
- Preserves original columns (`data_source`, `sequence`, `value`)
- Adds `prediction` column with interaction probability scores
- Saves to `results.csv` and uploads to GCS

## Model Parameters

### Required Parameters:
- **`--checkpoint-path`**: Path to PLM-interact checkpoint file (`pytorch_model.bin`)
  - Example: `../PLM-interact-650M-humanV12/pytorch_model.bin`
  - Available models on HuggingFace:
    - `danliu1226/PLM-interact-650M-humanV11`
    - `danliu1226/PLM-interact-650M-humanV12`
    - `danliu1226/PLM-interact-35M-humanV11`
    - `danliu1226/PLM-interact-650M-VH` (virus-human)
    - `danliu1226/PLM-interact-650M-Leakage-Free-Dataset`

- **`--offline-model-path`**: Path to offline ESM-2 model directory
  - Example: `../offline/test/`
  - Should contain the ESM-2 model files
  - Must end with `/` (script will add model name)

- **`--model-name`**: ESM-2 model name (must match checkpoint)
  - `esm2_t33_650M_UR50D` for 650M-parameter models
  - `esm2_t12_35M_UR50D` for 35M-parameter models

- **`--embedding-size`**: Embedding dimension
  - `1280` for `esm2_t33_650M_UR50D`
  - `480` for `esm2_t12_35M_UR50D`

- **`--max-length`**: Maximum sequence length
  - `1603` for 650M models (default)
  - Should be: combined length of both proteins + 3 (special tokens)

### Optional Parameters:
- **`--batch-size`**: Batch size for inference (default: 16)
- **`--device`**: Device for inference (default: `cuda:0`)
- **`--seed`**: Random seed (default: 2)
- **`--max-samples`**: Limit number of samples (for testing)

## Usage Examples

### Basic Usage:
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

### Using Custom GCS Path:
```bash
python eval_profluent_style/eval_profluent_style.py \
    --gcs-path gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi \
    --checkpoint-path ../PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi
```

### Testing with Limited Samples:
```bash
python eval_profluent_style/eval_profluent_style.py \
    --dataset-name alignment_skempi \
    --checkpoint-path ../PLM-interact-650M-humanV12/pytorch_model.bin \
    --offline-model-path ../offline/test/ \
    --model-name esm2_t33_650M_UR50D \
    --embedding-size 1280 \
    --max-length 1603 \
    --output-dir ./results/alignment_skempi \
    --max-samples 100 \
    --batch-size 8
```

## Downloading Models

### From HuggingFace:
```python
from huggingface_hub import snapshot_download
import os

# Download PLM-interact checkpoint
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

### ESM-2 Base Model:
The ESM-2 base model will be automatically downloaded by HuggingFace Transformers when you first run the script. To use offline models, download them separately:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save to offline path
offline_path = "../offline/test/esm2_t33_650M_UR50D"
model.save_pretrained(offline_path)
tokenizer.save_pretrained(offline_path)
```

## Output Format

The script generates two output files:

1. **`results.csv`**: Main results file with columns:
   - `data_source`: Original data source identifier
   - `sequence`: Original comma-separated protein pair
   - `value`: Ground truth value
   - `prediction`: PLM-interact interaction probability (0-1)

2. **`ppi_results.pkl`**: Pickle file with detailed results including:
   - `predictions`: NumPy array of predictions
   - `num_pairs`: Number of pairs processed
   - `dataset_name`: Dataset name
   - `dataset_gcs_path`: GCS path used
   - `num_samples`: Total samples processed

## Differences from Other Methods

### vs. MINT:
- **MINT**: Uses separate chain embeddings, computes cosine similarity
- **PLM-interact**: Jointly encodes pairs, uses binary classifier

### vs. ProteomeLM:
- **ProteomeLM**: Processes proteins independently, uses attention features
- **PLM-interact**: Processes pairs together, uses CLS token from joint encoding

## Troubleshooting

### Common Issues:

1. **Model mismatch**: Ensure `model_name` matches the checkpoint's base model
   - Check checkpoint README or HuggingFace page for base model info

2. **Max length too short**: If sequences are truncated, increase `--max-length`
   - Calculate: `max(protein1_length) + max(protein2_length) + 3`

3. **CUDA out of memory**: Reduce `--batch-size` (try 8 or 4)

4. **Offline model path**: Ensure path ends with `/` and contains the model files

5. **CSV format**: Ensure input CSV has `query` and `text` columns (not `Protein_Sequence_1`/`Protein_Sequence_2`)

## References

- PLM-interact paper: [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.11.05.622169v2)
- HuggingFace models: [danliu1226](https://huggingface.co/danliu1226)
- ESM-2 models: [facebook/esm2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)

