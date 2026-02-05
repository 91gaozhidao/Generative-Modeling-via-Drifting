#!/bin/bash
# =============================================================================
# Drifting Field Generative Model - Complete Reproduction Script
# =============================================================================
#
# This script orchestrates the full reproduction of the paper:
# "Generative Modeling via Drifting"
#
# It includes three main phases:
# 1. Data Preparation - Cache ImageNet images as VAE latents
# 2. MAE Pre-training - Pre-train the feature extractor using Masked Autoencoding
# 3. Drifting Training - Train the main DriftingDiT model
#
# Usage:
#   # Run full pipeline with default settings
#   ./run_reproduction.sh
#
#   # Run specific phase only
#   ./run_reproduction.sh --phase [data|mae|train|all]
#
#   # Run with dummy data (for testing)
#   ./run_reproduction.sh --dummy
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Data paths
# NOTE: IMAGENET_DIR must be set before running (unless using --dummy mode)
# Set via environment variable or --imagenet-dir argument
IMAGENET_DIR="${IMAGENET_DIR:-}"
CACHED_LATENTS_DIR="${CACHED_LATENTS_DIR:-./data/cached_latents}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MAE_WEIGHTS_DIR="${MAE_WEIGHTS_DIR:-./weights/mae}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

# MAE Pre-training
MAE_EPOCHS="${MAE_EPOCHS:-100}"
MAE_MASK_RATIO="${MAE_MASK_RATIO:-0.75}"
MAE_LR="${MAE_LR:-1.5e-4}"

# Main Training
TRAIN_EPOCHS="${TRAIN_EPOCHS:-100}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
MODEL_SIZE="${MODEL_SIZE:-base}"  # Options: small, base, large

# Dummy mode settings
USE_DUMMY=false
NUM_DUMMY_SAMPLES=1000

# =============================================================================
# Parse command line arguments
# =============================================================================

PHASE="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --dummy)
            USE_DUMMY=true
            shift
            ;;
        --imagenet-dir)
            IMAGENET_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --mae-epochs)
            MAE_EPOCHS="$2"
            shift 2
            ;;
        --train-epochs)
            TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --phase PHASE       Run specific phase: data, mae, train, or all (default: all)"
            echo "  --dummy             Use dummy data for testing pipeline"
            echo "  --imagenet-dir DIR  Path to ImageNet training data"
            echo "  --output-dir DIR    Output directory for checkpoints"
            echo "  --model-size SIZE   Model size: small, base, large (default: base)"
            echo "  --mae-epochs N      Number of MAE pre-training epochs (default: 100)"
            echo "  --train-epochs N    Number of main training epochs (default: 100)"
            echo "  --batch-size N      Batch size (default: 64)"
            echo "  --device DEVICE     Device to train on (default: cuda)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
    echo ""
}

check_directory() {
    if [[ ! -d "$1" ]]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    fi
}

# =============================================================================
# Phase 1: Data Preparation
# =============================================================================

run_data_preparation() {
    print_header "Phase 1: Data Preparation - Caching VAE Latents"
    
    check_directory "$CACHED_LATENTS_DIR"
    
    if [[ "$USE_DUMMY" == true ]]; then
        echo "Using dummy dataset with ${NUM_DUMMY_SAMPLES} samples..."
        python tools/cache_latents.py \
            --dummy \
            --num_dummy_samples "$NUM_DUMMY_SAMPLES" \
            --output_dir "$CACHED_LATENTS_DIR" \
            --device "$DEVICE"
    else
        # Validate IMAGENET_DIR is set
        if [[ -z "$IMAGENET_DIR" ]]; then
            echo "Error: IMAGENET_DIR is not set."
            echo ""
            echo "Please specify the ImageNet training data path using one of:"
            echo "  1. Environment variable: export IMAGENET_DIR=/path/to/imagenet/train"
            echo "  2. Command line: ./run_reproduction.sh --imagenet-dir /path/to/imagenet/train"
            echo ""
            echo "Or use --dummy mode for testing: ./run_reproduction.sh --dummy"
            exit 1
        fi
        
        echo "Caching ImageNet images from: $IMAGENET_DIR"
        echo "Output directory: $CACHED_LATENTS_DIR"
        
        if [[ ! -d "$IMAGENET_DIR" ]]; then
            echo "Error: ImageNet directory not found: $IMAGENET_DIR"
            echo "Please verify the path is correct."
            exit 1
        fi
        
        python tools/cache_latents.py \
            --data_dir "$IMAGENET_DIR" \
            --output_dir "$CACHED_LATENTS_DIR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --device "$DEVICE"
    fi
    
    echo "Data preparation complete!"
}

# =============================================================================
# Phase 2: MAE Pre-training
# =============================================================================

run_mae_pretraining() {
    print_header "Phase 2: MAE Pre-training - Training Feature Extractor"
    
    check_directory "$MAE_WEIGHTS_DIR"
    
    echo "Training MAE for ${MAE_EPOCHS} epochs..."
    echo "Mask ratio: ${MAE_MASK_RATIO}"
    echo "Learning rate: ${MAE_LR}"
    
    if [[ "$USE_DUMMY" == true ]]; then
        python tools/train_mae.py \
            --dummy \
            --num_dummy_samples "$NUM_DUMMY_SAMPLES" \
            --output_dir "$MAE_WEIGHTS_DIR" \
            --epochs "$MAE_EPOCHS" \
            --mask_ratio "$MAE_MASK_RATIO" \
            --learning_rate "$MAE_LR" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE"
    else
        python tools/train_mae.py \
            --data_dir "$CACHED_LATENTS_DIR" \
            --output_dir "$MAE_WEIGHTS_DIR" \
            --epochs "$MAE_EPOCHS" \
            --mask_ratio "$MAE_MASK_RATIO" \
            --learning_rate "$MAE_LR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --device "$DEVICE"
    fi
    
    echo ""
    echo "MAE pre-training complete!"
    echo "Encoder weights saved to: ${MAE_WEIGHTS_DIR}/best_encoder.pt"
}

# =============================================================================
# Phase 3: Drifting Training
# =============================================================================

run_drifting_training() {
    print_header "Phase 3: Drifting Training - Training Main Model"
    
    check_directory "$OUTPUT_DIR"
    
    MAE_CHECKPOINT="${MAE_WEIGHTS_DIR}/best_encoder.pt"
    
    echo "Training DriftingDiT (${MODEL_SIZE}) for ${TRAIN_EPOCHS} epochs..."
    echo "Learning rate: ${TRAIN_LR}"
    echo "MAE checkpoint: ${MAE_CHECKPOINT}"
    
    # Check if MAE checkpoint exists
    if [[ ! -f "$MAE_CHECKPOINT" ]]; then
        echo "Warning: MAE checkpoint not found at ${MAE_CHECKPOINT}"
        echo "Training without pre-trained feature extractor..."
        MAE_CHECKPOINT=""
    fi
    
    if [[ "$USE_DUMMY" == true ]]; then
        if [[ -n "$MAE_CHECKPOINT" ]]; then
            python train.py \
                --dummy \
                --num_dummy_samples "$NUM_DUMMY_SAMPLES" \
                --output_dir "$OUTPUT_DIR" \
                --model_size "$MODEL_SIZE" \
                --epochs "$TRAIN_EPOCHS" \
                --learning_rate "$TRAIN_LR" \
                --batch_size "$BATCH_SIZE" \
                --mae_checkpoint "$MAE_CHECKPOINT" \
                --device "$DEVICE"
        else
            python train.py \
                --dummy \
                --num_dummy_samples "$NUM_DUMMY_SAMPLES" \
                --output_dir "$OUTPUT_DIR" \
                --model_size "$MODEL_SIZE" \
                --epochs "$TRAIN_EPOCHS" \
                --learning_rate "$TRAIN_LR" \
                --batch_size "$BATCH_SIZE" \
                --device "$DEVICE"
        fi
    else
        if [[ -n "$MAE_CHECKPOINT" ]]; then
            python train.py \
                --data_dir "$CACHED_LATENTS_DIR" \
                --output_dir "$OUTPUT_DIR" \
                --model_size "$MODEL_SIZE" \
                --epochs "$TRAIN_EPOCHS" \
                --learning_rate "$TRAIN_LR" \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --mae_checkpoint "$MAE_CHECKPOINT" \
                --device "$DEVICE"
        else
            python train.py \
                --data_dir "$CACHED_LATENTS_DIR" \
                --output_dir "$OUTPUT_DIR" \
                --model_size "$MODEL_SIZE" \
                --epochs "$TRAIN_EPOCHS" \
                --learning_rate "$TRAIN_LR" \
                --batch_size "$BATCH_SIZE" \
                --num_workers "$NUM_WORKERS" \
                --device "$DEVICE"
        fi
    fi
    
    echo ""
    echo "Drifting training complete!"
    echo "Checkpoints saved to: ${OUTPUT_DIR}/checkpoints/"
}

# =============================================================================
# Main execution
# =============================================================================

print_header "Drifting Field Generative Model - Reproduction Pipeline"

echo "Configuration:"
echo "  Phase:         $PHASE"
echo "  Dummy mode:    $USE_DUMMY"
echo "  Model size:    $MODEL_SIZE"
echo "  Device:        $DEVICE"
echo "  Batch size:    $BATCH_SIZE"
echo ""

case $PHASE in
    data)
        run_data_preparation
        ;;
    mae)
        run_mae_pretraining
        ;;
    train)
        run_drifting_training
        ;;
    all)
        run_data_preparation
        run_mae_pretraining
        run_drifting_training
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: data, mae, train, all"
        exit 1
        ;;
esac

print_header "Pipeline Complete!"

echo "Summary:"
echo "  - Cached latents: ${CACHED_LATENTS_DIR}"
echo "  - MAE weights: ${MAE_WEIGHTS_DIR}"
echo "  - Model checkpoints: ${OUTPUT_DIR}/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Run FID evaluation:"
echo "     python tools/eval_fid.py --checkpoint ${OUTPUT_DIR}/checkpoints/best_model.pt"
echo ""
echo "  2. Generate samples:"
echo "     python generate.py --model_path ${OUTPUT_DIR}/checkpoints/best_model.pt"
echo ""
