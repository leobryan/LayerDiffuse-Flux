#!/bin/bash
# Example inference script for full fine-tuned model

# Configuration
BASE_MODEL="./models/flux_merged_base"  # Your merged base model
FINETUNED_TRANSFORMER="./style_lora_output/checkpoint-1000"  # Path to your trained checkpoint
TRANS_VAE="./models/TransparentVAE.pth"
OUTPUT_DIR="./inference_outputs"

# Generation parameters
PROMPT="glass bottle, high quality, artistic style"
WIDTH=1024
HEIGHT=1024
STEPS=50
GUIDANCE=3.5
SEED=42

# Run inference
python inference_full_model.py \
    --base_model "$BASE_MODEL" \
    --finetuned_transformer "$FINETUNED_TRANSFORMER" \
    --trans_vae "$TRANS_VAE" \
    --prompt "$PROMPT" \
    --output_dir "$OUTPUT_DIR" \
    --width $WIDTH \
    --height $HEIGHT \
    --steps $STEPS \
    --guidance $GUIDANCE \
    --seed $SEED

echo ""
echo "Done! Check $OUTPUT_DIR for generated images."
