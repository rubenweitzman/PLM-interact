#!/usr/bin/env python3
"""
PPI Prediction Evaluation Script (Profluent-style) for PLM-interact

Evaluates PLM-interact PPI prediction on datasets from dataset_to_eval.md.
Loads MDS datasets from GCS and runs PPI prediction pipeline using PLM-interact.

PLM-interact uses a binary classifier that takes protein pairs as input and outputs
interaction probability scores.

Usage:
    python eval_profluent_style/eval_profluent_style.py \
        --dataset-name alignment_skempi \
        --checkpoint-path ../PLM-interact-650M-humanV12/pytorch_model.bin \
        --offline-model-path ../offline/test/ \
        --model-name esm2_t33_650M_UR50D \
        --embedding-size 1280 \
        --max-length 1603 \
        --output-dir ./results/alignment_skempi
"""

import os
import sys
import logging
import click
import pickle
import subprocess
import shlex
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
from tqdm import tqdm

# Get the project root (parent of eval_profluent_style folder)
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root to path for PLM-interact imports
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import streaming (will be available if using pixi or if installed)
try:
    from streaming import StreamingDataset
except ImportError:
    logging.error("streaming package not found. Install with: pip install mosaicml-streaming")
    sys.exit(1)

# Import PLM-interact modules
try:
    from PLMinteract.inference.inference_PPI_singleGPU import CrossEncoder
except ImportError as e:
    logging.error(f"Failed to import PLM-interact modules: {e}")
    logging.error("Make sure PLM-interact is installed: pip install -e .")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset paths from dataset_to_eval.md
# Dataset paths - supports both MDS and CSV formats
DATASET_PATHS_MDS = {
    "alignment_skempi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_skempi",
    "alignment_mutational_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_mutational_ppi",
    "alignment_yeast_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_yeast_ppi_combined",
    "alignment_human_ppi_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_human_ppi_combined",
    "alignment_intact_ppi": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_intact_ppi",
    "validation_high_score_20_species": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/validation_high_score_20_species",
    "alignment_bindinggym_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_bindinggym_combined",
    "alignment_gold_combined": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/alignment_gold_combined",
    "human_validation_with_negatives": "gs://profluent-rweitzman/alignment/test_dataset_mds_round_2/human_validation_with_negatives",
}

# CSV paths (direct CSV format)
DATASET_PATHS_CSV = {
    "alignment_intact_covid": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_intact_covid.csv",
    "alignment_virus_human": "gs://profluent-rweitzman/alignment/test_dataset_csv_round_2/alignment_virus_human.csv",
}

# Combined lookup
DATASET_PATHS = {**DATASET_PATHS_MDS, **DATASET_PATHS_CSV}


def load_csv_dataset(csv_path: str, max_samples: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load CSV dataset from GCS or local path."""
    logger.info(f"Loading CSV dataset from: {csv_path}")
    
    if csv_path.startswith("gs://"):
        local_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
        logger.info(f"Downloading to: {local_csv}")
        cmd = f"gcloud storage cp {shlex.quote(csv_path)} {shlex.quote(local_csv)}"
        subprocess.run(shlex.split(cmd), check=True)
        csv_path = local_csv
    
    df = pd.read_csv(csv_path)
    logger.info(f"CSV contains {len(df)} rows, columns: {list(df.columns)}")
    
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        logger.info(f"Limited to {max_samples} samples")
    
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading samples"):
        samples.append({
            'sequence': row.get('sequence', ''),
            'value': float(row.get('value', 0.0)),
            'data_source': row.get('data_source', 'default')
        })
    
    return df, samples


def load_mds_dataset(gcs_path: str, max_samples: Optional[int] = None, local_cache_dir: Optional[str] = None) -> List[Dict]:
    """
    Load MDS dataset from GCS.
    
    Args:
        gcs_path: GCS path to MDS dataset
        max_samples: Maximum number of samples to load (None for all)
        local_cache_dir: Optional local directory for caching (auto-generated if None)
    
    Returns:
        List of samples, each with 'sequence' and 'value' fields
    """
    logger.info(f"Loading MDS dataset from: {gcs_path}")
    
    # Use temp directory for caching if not provided
    if local_cache_dir is None:
        local_cache_dir = tempfile.mkdtemp(prefix="mds_cache_")
        logger.info(f"Using temporary cache directory: {local_cache_dir}")
    
    dataset = StreamingDataset(
        remote=gcs_path,
        local=local_cache_dir,
        batch_size=1000,
        shuffle=False,
        num_canonical_nodes=1,
        download_timeout=600,
    )
    
    total_samples = len(dataset)
    logger.info(f"Dataset contains {total_samples} samples")
    
    # Determine how many samples to load
    num_to_load = min(max_samples, total_samples) if max_samples else total_samples
    
    samples = []
    with tqdm(total=num_to_load, desc="Loading samples", unit="samples") as pbar:
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            samples.append({
                'sequence': sample.get('sequence', ''),
                'value': float(sample.get('value', 0.0)),
                'data_source': sample.get('data_source', 'default')
            })
            pbar.update(1)
    
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def extract_protein_pairs(samples: List[Dict]) -> List[Dict]:
    """
    Extract protein pairs from samples.
    
    PLM-interact expects pairs in CSV format with 'query' and 'text' columns.
    
    Returns:
        List of dicts with 'query' (protein1) and 'text' (protein2)
    """
    pairs = []
    
    for i, sample in enumerate(tqdm(samples, desc="Extracting protein pairs", unit="samples")):
        seq = sample['sequence']
        
        # Split by comma (always the separator in these datasets)
        # Use split(',', 1) to split only on first comma in case sequence contains commas
        parts = seq.split(',', 1)
        if len(parts) == 2:
            seq1, seq2 = parts[0].strip(), parts[1].strip()
            if seq1 and seq2:
                pairs.append({
                    'query': seq1,
                    'text': seq2
                })
            else:
                logger.warning(f"Empty sequence in sample {i}")
                pairs.append({
                    'query': '',
                    'text': ''
                })
        else:
            logger.warning(f"Sample {i} does not contain comma-separated pair: {seq[:50]}...")
            pairs.append({
                'query': '',
                'text': ''
            })
    
    logger.info(f"Extracted {len(pairs)} protein pairs from {len(samples)} samples")
    
    return pairs


def create_csv_file(pairs: List[Dict], output_path: str) -> str:
    """Create a CSV file from pairs in PLM-interact format (query, text columns)."""
    df = pd.DataFrame(pairs)
    df.to_csv(output_path, index=False)
    logger.info(f"Created CSV file: {output_path} with {len(pairs)} pairs")
    return output_path


def run_ppi_prediction(
    csv_file: str,
    checkpoint_path: str,
    offline_model_path: str,
    model_name: str,
    embedding_size: int,
    max_length: int,
    output_dir: Path,
    device: str = "cuda:0",
    batch_size: int = 16,
    seed: int = 2,
) -> Dict:
    """
    Run PLM-interact PPI prediction pipeline.
    
    Args:
        csv_file: Path to CSV file with 'query' and 'text' columns
        checkpoint_path: Path to PLM-interact checkpoint (pytorch_model.bin)
        offline_model_path: Path to offline ESM-2 model directory (should end with /)
        model_name: ESM-2 model name (e.g., 'esm2_t33_650M_UR50D')
        embedding_size: Embedding size (1280 for 650M, 480 for 35M)
        max_length: Maximum sequence length
        output_dir: Output directory for results
        device: Device for inference (not used directly, CrossEncoder uses cuda:0)
        batch_size: Batch size for inference
        seed: Random seed
    
    Returns:
        Dictionary with results including predictions
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if checkpoint exists, if not try to download from HuggingFace
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        # Check if it's a HuggingFace repo ID
        if '/' in checkpoint_path and not checkpoint_path.startswith('/') and not checkpoint_path.startswith('.'):
            logger.info(f"Checkpoint not found locally: {checkpoint_path}")
            logger.info("Attempting to download from HuggingFace...")
            try:
                from huggingface_hub import snapshot_download
                # Extract repo ID (assuming format like "danliu1226/PLM-interact-650M-humanV12")
                if checkpoint_path.startswith('danliu1226/'):
                    repo_id = checkpoint_path
                    local_dir = f"./models/{checkpoint_path.split('/')[-1]}"
                    logger.info(f"Downloading {repo_id} to {local_dir}...")
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                    )
                    checkpoint_path = str(Path(local_dir) / "pytorch_model.bin")
                    logger.info(f"✓ Downloaded checkpoint to: {checkpoint_path}")
                else:
                    logger.error(f"Unknown HuggingFace repo format: {checkpoint_path}")
                    logger.error("Please provide a local path or use format: danliu1226/PLM-interact-650M-humanV12")
                    sys.exit(1)
            except ImportError:
                logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
                logger.error(f"Or download model manually to: {checkpoint_path}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                logger.error(f"Please download manually to: {checkpoint_path}")
                sys.exit(1)
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            logger.error("Please download the model first:")
            logger.error("  python download_model.py --model humanV12")
            sys.exit(1)
    
    logger.info(f"Loading PLM-interact model from checkpoint: {checkpoint_path}")
    
    # Create model path
    # If offline_model_path is empty or not provided, use model_name directly (HuggingFace)
    if offline_model_path and offline_model_path.strip():
        # Ensure offline_model_path ends with / for concatenation
        if not offline_model_path.endswith('/'):
            offline_model_path = offline_model_path + '/'
        model_path = offline_model_path + model_name
    else:
        # Use HuggingFace model name directly
        model_path = model_name
        offline_model_path = ""  # Ensure it's empty string
    
    # Initialize CrossEncoder
    trainer = CrossEncoder(
        model_name=model_path,
        num_labels=1,
        max_length=max_length,
        embedding_size=embedding_size,
        checkpoint=checkpoint_path
    )
    
    # Create args object matching the expected format
    class Args:
        def __init__(self):
            self.resume_from_checkpoint = checkpoint_path
            self.offline_model_path = offline_model_path
            self.seed = seed
            self.batch_size_val = batch_size
            self.test_filepath = csv_file
            self.output_filepath = str(output_dir) + '/'  # Must end with /
            self.model_name = model_name
            self.embedding_size = embedding_size
            self.max_length = max_length
    
    args = Args()
    
    # Run inference
    logger.info("Running PLM-interact inference...")
    trainer.inference(
        args,
        batch_size_val=batch_size,
        output_path=args.output_filepath,
    )
    
    # Load predictions
    pred_file = output_dir / "pred_scores.csv"
    if pred_file.exists():
        predictions = pd.read_csv(pred_file, header=None).values.flatten()
        logger.info(f"Loaded {len(predictions)} predictions")
        logger.info(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
    else:
        logger.error(f"Prediction file not found: {pred_file}")
        predictions = np.array([])
    
    return {
        'predictions': predictions,
        'num_pairs': len(predictions),
    }


@click.command()
@click.option(
    "--dataset-name",
    type=str,
    help="Dataset name from dataset_to_eval.md (e.g., 'alignment_skempi')"
)
@click.option(
    "--gcs-path",
    type=str,
    help="GCS path to MDS dataset (overrides dataset-name if provided)"
)
@click.option(
    "--csv-path",
    type=str,
    help="GCS or local path to CSV file (format: sequence,value,data_source). Overrides --gcs-path"
)
@click.option(
    "--checkpoint-path",
    type=str,
    required=True,
    help="Path to PLM-interact checkpoint (e.g., '../PLM-interact-650M-humanV12/pytorch_model.bin'). "
         "Recommended: PLM-interact-650M-humanV12 for general human PPI prediction. "
         "Other options: PLM-interact-650M-humanV11, PLM-interact-35M-humanV11 (faster), "
         "PLM-interact-650M-VH (virus-human), PLM-interact-650M-Leakage-Free-Dataset"
)
@click.option(
    "--offline-model-path",
    type=str,
    required=True,
    help="Path to offline ESM-2 model directory (e.g., '../offline/test/')"
)
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="ESM-2 model name. Must match checkpoint: "
         "'esm2_t33_650M_UR50D' for 650M models (humanV11, humanV12, VH, etc.), "
         "'esm2_t12_35M_UR50D' for 35M models (humanV11)"
)
@click.option(
    "--embedding-size",
    type=int,
    required=True,
    help="Embedding size. Must match model: "
         "1280 for esm2_t33_650M_UR50D (650M models), "
         "480 for esm2_t12_35M_UR50D (35M models)"
)
@click.option(
    "--max-length",
    type=int,
    required=True,
    help="Maximum sequence length (e.g., 1603 for 650M model)"
)
@click.option(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for results"
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum number of samples to process (for testing, None = all)"
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help="Device for PLM-interact model (default: cuda:0)"
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for inference (default: 16)"
)
@click.option(
    "--seed",
    type=int,
    default=2,
    help="Random seed (default: 2)"
)
def main(
    dataset_name: Optional[str],
    gcs_path: Optional[str],
    csv_path: Optional[str],
    checkpoint_path: str,
    offline_model_path: str,
    model_name: str,
    embedding_size: int,
    max_length: int,
    output_dir: str,
    max_samples: Optional[int],
    device: str,
    batch_size: int,
    seed: int,
) -> None:
    """Run PLM-interact PPI prediction evaluation on MDS or CSV dataset."""
    
    # Determine data source (CSV takes priority)
    use_csv = False
    original_df = None
    
    if csv_path:
        data_path = csv_path
        use_csv = True
    elif gcs_path:
        data_path = gcs_path
        use_csv = gcs_path.endswith('.csv')
    elif dataset_name and dataset_name in DATASET_PATHS:
        data_path = DATASET_PATHS[dataset_name]
        use_csv = dataset_name in DATASET_PATHS_CSV or data_path.endswith('.csv')
    else:
        logger.error(f"Must provide --csv-path, --gcs-path, or --dataset-name (one of: {list(DATASET_PATHS.keys())})")
        sys.exit(1)
    
    logger.info("="*80)
    logger.info("PLM-interact PPI Prediction Evaluation")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name or 'custom'}")
    logger.info(f"Data Path: {data_path}")
    logger.info(f"Format: {'CSV' if use_csv else 'MDS'}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Embedding Size: {embedding_size}")
    logger.info(f"Max Length: {max_length}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load dataset (CSV or MDS)
    if use_csv:
        logger.info("\n[Step 1/4] Loading CSV dataset...")
        original_df, samples = load_csv_dataset(data_path, max_samples=max_samples)
    else:
        logger.info("\n[Step 1/4] Loading MDS dataset...")
        samples = load_mds_dataset(data_path, max_samples=max_samples)
    
    # Step 2: Extract protein pairs
    logger.info("\n[Step 2/4] Extracting protein pairs...")
    pairs = extract_protein_pairs(samples)
    
    if len(pairs) == 0:
        logger.error("No protein pairs extracted from samples!")
        sys.exit(1)
    
    # Step 3: Create CSV file
    logger.info("\n[Step 3/4] Creating CSV file...")
    csv_file = str(output_path / "protein_pairs.csv")
    create_csv_file(pairs, csv_file)
    
    # Step 4: Run PPI prediction
    logger.info("\n[Step 4/4] Running PLM-interact PPI prediction...")
    results = run_ppi_prediction(
        csv_file=csv_file,
        checkpoint_path=checkpoint_path,
        offline_model_path=offline_model_path,
        model_name=model_name,
        embedding_size=embedding_size,
        max_length=max_length,
        output_dir=output_path,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    
    # Step 5: Create CSV output with predictions
    logger.info("\n[Step 5/5] Creating CSV output with predictions...")
    
    predictions = results['predictions']
    
    # If we loaded from CSV, add column to original DataFrame
    if use_csv and original_df is not None:
        logger.info("Adding prediction column to original CSV...")
        output_df = original_df.copy()
        
        if len(predictions) == len(output_df):
            output_df['plm_interact_prediction'] = predictions
        else:
            logger.warning(f"Prediction count mismatch: {len(predictions)} vs {len(output_df)} rows")
            output_df['plm_interact_prediction'] = np.nan
            output_df.loc[:len(predictions)-1, 'plm_interact_prediction'] = predictions
    else:
        # Build output rows from samples (MDS format)
        output_rows = []
        for i, sample in enumerate(tqdm(samples, desc="Creating output", unit="samples")):
            if i < len(predictions):
                prediction_score = float(predictions[i])
            else:
                logger.warning(f"Could not find prediction for sample {i}")
                prediction_score = np.nan
            
            row = {
                'data_source': sample.get('data_source', ''),
                'sequence': sample['sequence'],
                'value': sample['value'],
                'plm_interact_prediction': prediction_score,
            }
            output_rows.append(row)
        
        output_df = pd.DataFrame(output_rows)
    
    csv_output_file = output_path / "results.csv"
    output_df.to_csv(csv_output_file, index=False)
    logger.info(f"Saved CSV results to {csv_output_file}")
    logger.info(f"CSV contains {len(output_df)} rows with columns: {list(output_df.columns)}")
    
    # Also save pickle for detailed analysis
    results_file = output_path / "ppi_results.pkl"
    logger.info(f"\nSaving detailed results to {results_file}")
    
    # Add metadata to results
    results['dataset_name'] = dataset_name or 'custom'
    results['data_path'] = data_path
    results['num_samples'] = len(samples)
    results['num_pairs'] = len(pairs)
    
    with open(results_file, "wb") as f:
        pickle.dump(results, f)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Complete!")
    logger.info("="*80)
    logger.info(f"Total samples processed: {len(samples)}")
    logger.info(f"Total pairs: {len(pairs)}")
    
    # Log predictions
    if 'plm_interact_prediction' in output_df.columns and not output_df['plm_interact_prediction'].isna().all():
        valid_preds = output_df['plm_interact_prediction'].dropna()
        logger.info(f"PLM-interact prediction range: {valid_preds.min():.4f} - {valid_preds.max():.4f}")
        logger.info(f"PLM-interact prediction mean: {valid_preds.mean():.4f}")
    
    logger.info(f"CSV results saved to: {csv_output_file}")
    logger.info(f"Detailed results saved to: {results_file}")
    
    # Upload results to GCS
    gcs_bucket = "profluent-rweitzman"
    method_name = "plm_interact"
    gcs_base_path = f"gs://{gcs_bucket}/baseline_results/{method_name}/{dataset_name or 'custom'}"
    
    logger.info(f"\nUploading results to GCS: {gcs_base_path}")
    try:
        # Upload CSV file
        csv_gcs_path = f"{gcs_base_path}/results.csv"
        logger.info(f"Uploading {csv_output_file} -> {csv_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(csv_output_file))} {shlex.quote(csv_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded CSV to {csv_gcs_path}")
        
        # Upload pickle file (optional, but useful for detailed analysis)
        pkl_gcs_path = f"{gcs_base_path}/ppi_results.pkl"
        logger.info(f"Uploading {results_file} -> {pkl_gcs_path}")
        cmd = f"gcloud storage cp {shlex.quote(str(results_file))} {shlex.quote(pkl_gcs_path)}"
        subprocess.run(shlex.split(cmd), check=True)
        logger.info(f"✓ Successfully uploaded pickle to {pkl_gcs_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to upload to GCS: {e}")
        logger.error("Results are still saved locally")
    except Exception as e:
        logger.error(f"Unexpected error uploading to GCS: {e}")
        logger.error("Results are still saved locally")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()

