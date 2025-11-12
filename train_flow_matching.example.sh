#!/bin/bash
# Example training script with Flow Matching loss (latent-space, official implementation)

# Model paths
BASE_MODEL="./models/flux_merged_base"
TRANS_VAE="./models/TransparentVAE.pth"
DATA_DIR="./training_data"
OUTPUT_DIR="./style_lora_output_flow"

# Training settings
RESOLUTION=1024
ASPECT_RATIO="square"
BATCH_SIZE=4
GRADIENT_ACCUM=4
MAX_STEPS=5000
LEARNING_RATE=1e-4

# LoRA settings
LORA_RANK=32
LORA_ALPHA=32

# Flow Matching settings
WEIGHTING_SCHEME="none"  # none, sigma_sqrt, logit_normal, mode, cosmap
GUIDANCE_SCALE=3.5

# Run training
accelerate launch --mixed_precision=bf16 train_style_lora_flow.py \
    --base_model "$BASE_MODEL" \
    --trans_vae "$TRANS_VAE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --aspect_ratio_type "$ASPECT_RATIO" \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --max_train_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --weighting_scheme $WEIGHTING_SCHEME \
    --guidance_scale $GUIDANCE_SCALE \
    --gradient_checkpointing \
    --random_flip \
    --mixed_precision bf16 \
    --logging_steps 10 \
    --checkpointing_steps 500 \
    --seed 42

echo "Training completed! Output saved to: $OUTPUT_DIR"
