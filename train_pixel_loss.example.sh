#!/bin/bash
# Example training script with pixel-space loss (image reconstruction)

# Model paths
BASE_MODEL="./models/flux_merged_base"  # Your merged base model
TRANS_VAE="./models/TransparentVAE.pth"
DATA_DIR="./training_data"
OUTPUT_DIR="./style_lora_output_pixel"

# Training settings
RESOLUTION=1024
ASPECT_RATIO="square"  # square, portrait, landscape, or auto
BATCH_SIZE=1  # Keep small - pixel loss needs more memory
GRADIENT_ACCUM=8  # Increase to compensate for small batch size
MAX_STEPS=3000

# Loss settings
RGB_WEIGHT=1.0
ALPHA_WEIGHT=1.0
LOSS_TYPE="l2"  # l1, l2, or huber

# LoRA settings
LORA_RANK=32
LORA_ALPHA=32
LEARNING_RATE=1e-4

# Generation settings (per training step)
NUM_INFERENCE_STEPS=28  # Fewer steps = faster training, but less accurate
GUIDANCE_SCALE=3.5

# Run training
accelerate launch --mixed_precision=bf16 train_style_lora_pixel_loss.py \
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
    --rgb_loss_weight $RGB_WEIGHT \
    --alpha_loss_weight $ALPHA_WEIGHT \
    --loss_type $LOSS_TYPE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --guidance_scale $GUIDANCE_SCALE \
    --gradient_checkpointing \
    --random_flip \
    --mixed_precision bf16 \
    --logging_steps 10 \
    --checkpointing_steps 500 \
    --seed 42

echo "Training completed! Output saved to: $OUTPUT_DIR"
