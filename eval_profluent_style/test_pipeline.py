#!/usr/bin/env python3
"""
Test PLM-interact pipeline with a small example dataset.

This script creates a small test dataset and runs the PLM-interact inference
pipeline to verify everything works correctly.
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
import click

# Add parent directory to path for PLM-interact imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import functions from eval_profluent_style module
# We import directly since we're in the same directory
import importlib.util
eval_script_path = Path(__file__).parent / "eval_profluent_style.py"
spec = importlib.util.spec_from_file_location("eval_profluent_style", eval_script_path)
eval_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_module)

extract_protein_pairs = eval_module.extract_protein_pairs
create_csv_file = eval_module.create_csv_file
run_ppi_prediction = eval_module.run_ppi_prediction


def create_test_data(output_file: str, num_samples: int = 10):
    """Create a small test CSV file with protein pairs."""
    
    # Example protein sequences (short for testing)
    test_sequences = [
        ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSMLLNDGILM", 
         "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSMLLNDGILM"),
        ("MKTIIALSYIFCLVFA", 
         "MKLLVVVFCLGIAPSFHQ"),
        ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSMLLNDGILM",
         "MKLLVVVFCLGIAPSFHQ"),
    ]
    
    # Create test samples
    samples = []
    for i in range(num_samples):
        seq1, seq2 = test_sequences[i % len(test_sequences)]
        samples.append({
            'sequence': f"{seq1},{seq2}",
            'value': float(i % 2),  # Alternating labels
            'data_source': f'test_{i}'
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(samples)
    df.to_csv(output_file, index=False)
    
    print(f"Created test dataset with {len(samples)} samples")
    print(f"Saved to: {output_file}")
    
    return samples


@click.command()
@click.option(
    "--checkpoint-path",
    type=str,
    required=True,
    help="Path to PLM-interact checkpoint (pytorch_model.bin)"
)
@click.option(
    "--offline-model-path",
    type=str,
    required=True,
    help="Path to offline ESM-2 model directory"
)
@click.option(
    "--model-name",
    type=str,
    required=True,
    help="ESM-2 model name (e.g., 'esm2_t33_650M_UR50D')"
)
@click.option(
    "--embedding-size",
    type=int,
    required=True,
    help="Embedding size (1280 for 650M, 480 for 35M)"
)
@click.option(
    "--max-length",
    type=int,
    default=1603,
    help="Maximum sequence length (default: 1603)"
)
@click.option(
    "--num-samples",
    type=int,
    default=5,
    help="Number of test samples to generate (default: 5)"
)
@click.option(
    "--batch-size",
    type=int,
    default=2,
    help="Batch size for testing (default: 2)"
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help="Device for inference (default: cuda:0)"
)
def test_pipeline(
    checkpoint_path: str,
    offline_model_path: str,
    model_name: str,
    embedding_size: int,
    max_length: int,
    num_samples: int,
    batch_size: int,
    device: str,
):
    """Test PLM-interact pipeline with a small example dataset."""
    
    print("="*80)
    print("PLM-interact Pipeline Test")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}")
    print(f"Embedding size: {embedding_size}")
    print(f"Max length: {max_length}")
    print(f"Test samples: {num_samples}")
    print("="*80)
    
    # Check if checkpoint exists
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        print(f"\n✗ Error: Checkpoint not found: {checkpoint_path}")
        print("\nPlease download the model first:")
        print("  python download_model.py --model humanV12")
        sys.exit(1)
    
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Create test data
        print("\n[Step 1/4] Creating test dataset...")
        test_csv = temp_path / "test_data.csv"
        samples = create_test_data(str(test_csv), num_samples=num_samples)
        
        # Step 2: Extract protein pairs
        print("\n[Step 2/4] Extracting protein pairs...")
        pairs = extract_protein_pairs(samples)
        print(f"✓ Extracted {len(pairs)} pairs")
        
        # Step 3: Create CSV file for PLM-interact
        print("\n[Step 3/4] Creating PLM-interact CSV format...")
        plm_csv = temp_path / "protein_pairs.csv"
        create_csv_file(pairs, str(plm_csv))
        print(f"✓ Created CSV file: {plm_csv}")
        
        # Step 4: Run inference
        print("\n[Step 4/4] Running PLM-interact inference...")
        try:
            results = run_ppi_prediction(
                csv_file=str(plm_csv),
                checkpoint_path=str(checkpoint_file),
                offline_model_path=offline_model_path,
                model_name=model_name,
                embedding_size=embedding_size,
                max_length=max_length,
                output_dir=temp_path,
                device=device,
                batch_size=batch_size,
                seed=2,
            )
            
            predictions = results['predictions']
            
            print("\n" + "="*80)
            print("✓ Test Successful!")
            print("="*80)
            print(f"Processed {len(predictions)} protein pairs")
            print(f"Prediction range: {predictions.min():.4f} - {predictions.max():.4f}")
            print(f"Prediction mean: {predictions.mean():.4f}")
            
            # Show sample predictions
            print("\nSample predictions:")
            for i, (sample, pred) in enumerate(zip(samples[:5], predictions[:5])):
                seq_preview = sample['sequence'][:50] + "..." if len(sample['sequence']) > 50 else sample['sequence']
                print(f"  {i+1}. {seq_preview}")
                print(f"     Prediction: {pred:.4f}")
            
            print("\n" + "="*80)
            print("Pipeline test completed successfully!")
            print("="*80)
            print("\nYou can now run the full evaluation:")
            print(f"  python eval_profluent_style.py \\")
            print(f"    --dataset-name alignment_skempi \\")
            print(f"    --checkpoint-path {checkpoint_path} \\")
            print(f"    --offline-model-path {offline_model_path} \\")
            print(f"    --model-name {model_name} \\")
            print(f"    --embedding-size {embedding_size} \\")
            print(f"    --max-length {max_length} \\")
            print(f"    --output-dir ./results/test")
            
        except Exception as e:
            print(f"\n✗ Error during inference: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    test_pipeline()

