#!/bin/bash
# Optimized training config for 24GB GPU (RTX 3090/4090/A5000)

BASE_MODEL="./models/flux_merged_base"
TRANS_VAE="./models/TransparentVAE.pth"
DATA_DIR="./training_data"
OUTPUT_DIR="./style_lora_output_flow_optimized"

# Training settings optimized for 24GB
RESOLUTION=1024
BATCH_SIZE=1           # Small batch size
GRADIENT_ACCUM=16      # Compensate with large accumulation
MAX_STEPS=5000
LEARNING_RATE=1e-4

# LoRA settings
LORA_RANK=32
LORA_ALPHA=32

# Flow Matching settings
WEIGHTING_SCHEME="none"
GUIDANCE_SCALE=3.5

# Run training with MAXIMUM memory optimization
accelerate launch --mixed_precision=bf16 train_style_lora_flow_optimized.py \
    --base_model "$BASE_MODEL" \
    --trans_vae "$TRANS_VAE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --resolution $RESOLUTION \
    --aspect_ratio_type square \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUM \
    --max_train_steps $MAX_STEPS \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --weighting_scheme $WEIGHTING_SCHEME \
    --guidance_scale $GUIDANCE_SCALE \
    --gradient_checkpointing \
    --cache_latents \
    --cache_text_embeddings \
    --offload_text_encoder_to_cpu \
    --use_8bit_adam \
    --mixed_precision bf16 \
    --dataloader_num_workers 0 \
    --logging_steps 10 \
    --checkpointing_steps 500 \
    --log_memory_usage \
    --seed 42

echo ""
echo "Training completed! Output saved to: $OUTPUT_DIR"
echo ""
echo "Memory optimization strategies used:"
echo "  ✓ Latent caching (VAE not needed during training)"
echo "  ✓ Text embedding caching (Text encoders not needed)"
echo "  ✓ Text encoders offloaded to CPU"
echo "  ✓ 8-bit Adam optimizer"
echo "  ✓ Gradient checkpointing"
echo "  ✓ Mixed precision (bf16)"
echo ""
echo "Expected memory usage: ~18-20GB"
