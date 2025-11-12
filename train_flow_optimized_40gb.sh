#!/bin/bash
# Optimized training config for 40GB GPU (A100/A6000)
# Balance between speed and memory efficiency

BASE_MODEL="./models/flux_merged_base"
TRANS_VAE="./models/TransparentVAE.pth"
DATA_DIR="./training_data"
OUTPUT_DIR="./style_lora_output_flow_optimized"

# Training settings optimized for 40GB
RESOLUTION=1024
BATCH_SIZE=2           # Can afford larger batch
GRADIENT_ACCUM=8       # Less accumulation needed
MAX_STEPS=5000
LEARNING_RATE=1e-4

# LoRA settings
LORA_RANK=32
LORA_ALPHA=32

# Flow Matching settings
WEIGHTING_SCHEME="none"
GUIDANCE_SCALE=3.5

# Run training with balanced optimization
# Not all optimizations needed on 40GB, but caching still helps speed
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
echo "  ✓ Latent caching (speeds up training!)"
echo "  ✓ Text embedding caching (speeds up training!)"
echo "  ✓ Gradient checkpointing"
echo "  ✓ Mixed precision (bf16)"
echo ""
echo "Expected memory usage: ~22-25GB"
echo "Note: Caching actually SPEEDS UP training on large GPUs!"
