#!/bin/bash
# Quick test script to download model and test pipeline

set -e

echo "=========================================="
echo "PLM-interact Quick Test"
echo "=========================================="

# Default model (can be overridden)
MODEL=${1:-humanV12}
MODEL_DIR="./models/PLM-interact-650M-${MODEL}"

if [ "$MODEL" == "35M" ]; then
    MODEL_DIR="./models/PLM-interact-35M-humanV11"
fi

CHECKPOINT="${MODEL_DIR}/pytorch_model.bin"

# Step 1: Download model
echo ""
echo "Step 1: Downloading model..."
python download_model.py --model ${MODEL} --output-dir ${MODEL_DIR}

# Step 2: Set parameters based on model
if [ "$MODEL" == "35M" ]; then
    MODEL_NAME="esm2_t12_35M_UR50D"
    EMBEDDING_SIZE=480
else
    MODEL_NAME="esm2_t33_650M_UR50D"
    EMBEDDING_SIZE=1280
fi

# Step 3: Test pipeline
echo ""
echo "Step 2: Testing pipeline..."
python test_pipeline.py \
    --checkpoint-path ${CHECKPOINT} \
    --offline-model-path ../offline/test/ \
    --model-name ${MODEL_NAME} \
    --embedding-size ${EMBEDDING_SIZE} \
    --max-length 1603 \
    --num-samples 5 \
    --batch-size 2

echo ""
echo "=========================================="
echo "✓ Quick test completed!"
echo "=========================================="

