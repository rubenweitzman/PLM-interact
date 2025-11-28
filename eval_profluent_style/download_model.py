#!/usr/bin/env python3
"""
Download PLM-interact models from HuggingFace.

This script downloads the recommended PLM-interact-650M-humanV12 model
or any other model specified.
"""

import os
import sys
import click
from pathlib import Path
from huggingface_hub import snapshot_download

# Available models
AVAILABLE_MODELS = {
    "humanV12": {
        "repo_id": "danliu1226/PLM-interact-650M-humanV12",
        "description": "Recommended: Trained on STRING V12 (most recent)",
        "base_model": "esm2_t33_650M_UR50D",
        "embedding_size": 1280,
        "max_length": 1603,
    },
    "humanV11": {
        "repo_id": "danliu1226/PLM-interact-650M-humanV11",
        "description": "Trained on cross-species dataset",
        "base_model": "esm2_t33_650M_UR50D",
        "embedding_size": 1280,
        "max_length": 1603,
    },
    "35M": {
        "repo_id": "danliu1226/PLM-interact-35M-humanV11",
        "description": "Smaller, faster model",
        "base_model": "esm2_t12_35M_UR50D",
        "embedding_size": 480,
        "max_length": 1603,
    },
    "VH": {
        "repo_id": "danliu1226/PLM-interact-650M-VH",
        "description": "Virus-human interactions",
        "base_model": "esm2_t33_650M_UR50D",
        "embedding_size": 1280,
        "max_length": 1603,
    },
    "leakage-free": {
        "repo_id": "danliu1226/PLM-interact-650M-Leakage-Free-Dataset",
        "description": "Leakage-free evaluation",
        "base_model": "esm2_t33_650M_UR50D",
        "embedding_size": 1280,
        "max_length": 1603,
    },
    "mutation": {
        "repo_id": "danliu1226/PLM-interact-650M-Mutation",
        "description": "Mutation effect prediction",
        "base_model": "esm2_t33_650M_UR50D",
        "embedding_size": 1280,
        "max_length": 1603,
    },
}


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(AVAILABLE_MODELS.keys())),
    default="humanV12",
    help="Model to download (default: humanV12 - recommended)"
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory (default: ./models/{model_name})"
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-download even if model exists"
)
def download_model(model: str, output_dir: str, force: bool):
    """Download PLM-interact model from HuggingFace."""
    
    model_info = AVAILABLE_MODELS[model]
    
    # Determine output directory
    if output_dir is None:
        output_dir = f"./models/PLM-interact-650M-{model}" if model != "35M" else "./models/PLM-interact-35M-humanV11"
    
    output_path = Path(output_dir)
    
    # Check if model already exists
    checkpoint_file = output_path / "pytorch_model.bin"
    if checkpoint_file.exists() and not force:
        print(f"✓ Model already exists at {output_path}")
        print(f"  Checkpoint: {checkpoint_file}")
        print(f"  Use --force to re-download")
        print_config(model, model_info, output_path)
        return
    
    print("="*80)
    print(f"Downloading PLM-interact Model: {model}")
    print("="*80)
    print(f"Repository: {model_info['repo_id']}")
    print(f"Description: {model_info['description']}")
    print(f"Output directory: {output_path}")
    print(f"Base model: {model_info['base_model']}")
    print(f"Embedding size: {model_info['embedding_size']}")
    print(f"Max length: {model_info['max_length']}")
    print("="*80)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\nDownloading from HuggingFace...")
        print(f"This may take several minutes (model size: ~2.6 GB for 650M, ~140 MB for 35M)...")
        
        snapshot_download(
            repo_id=model_info['repo_id'],
            local_dir=str(output_path),
            local_dir_use_symlinks=False,
            force_download=force,
        )
        
        print("\n" + "="*80)
        print("✓ Download Complete!")
        print("="*80)
        print(f"Model saved to: {output_path}")
        print(f"Checkpoint file: {checkpoint_file}")
        
        print_config(model, model_info, output_path)
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~3 GB for 650M models)")
        print("3. Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def print_config(model: str, model_info: dict, output_path: Path):
    """Print configuration for using the downloaded model."""
    checkpoint_path = output_path / "pytorch_model.bin"
    
    print("\n" + "="*80)
    print("Usage Example:")
    print("="*80)
    print(f"""
python eval_profluent_style/eval_profluent_style.py \\
    --dataset-name alignment_skempi \\
    --checkpoint-path {checkpoint_path} \\
    --offline-model-path ../offline/test/ \\
    --model-name {model_info['base_model']} \\
    --embedding-size {model_info['embedding_size']} \\
    --max-length {model_info['max_length']} \\
    --output-dir ./results/test
""")
    print("="*80)


if __name__ == "__main__":
    download_model()

