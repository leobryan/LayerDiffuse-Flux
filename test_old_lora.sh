#!/bin/bash
# Test script for old format LoRA weights (model.safetensors / adapter_model.safetensors)

# Configuration
BASE_MODEL="./models/flux_merged_base"  # Your merged base model
LORA_CHECKPOINT="./style_lora_output/checkpoint-1000"  # Path to checkpoint with adapter_model.safetensors or model.safetensors
TRANS_VAE="./models/TransparentVAE.pth"
OUTPUT_DIR="./inference_outputs"

# Generation parameters
PROMPT="glass bottle, high quality, artistic style"
WIDTH=1024
HEIGHT=1024
STEPS=50
GUIDANCE=3.5
SEED=42

echo "================================================================"
echo "Testing Old Format LoRA Inference"
echo "================================================================"
echo ""
echo "Checkpoint directory: $LORA_CHECKPOINT"
echo ""
echo "Files in checkpoint:"
ls -lh "$LORA_CHECKPOINT" 2>/dev/null || echo "Checkpoint directory not found!"
echo ""
echo "================================================================"
echo ""

# Run inference
python inference_old_lora.py \
    --base_model "$BASE_MODEL" \
    --lora_checkpoint "$LORA_CHECKPOINT" \
    --trans_vae "$TRANS_VAE" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --steps $STEPS \
    --guidance $GUIDANCE \
    --seed $SEED \
    --output_name "old_lora_test"

echo ""
echo "Done! Check $OUTPUT_DIR for generated images."
