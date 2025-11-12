#!/bin/bash
# Aggressive memory optimization for 16GB GPU (RTX 4080/4070 Ti)

BASE_MODEL="./models/flux_merged_base"
TRANS_VAE="./models/TransparentVAE.pth"
DATA_DIR="./training_data"
OUTPUT_DIR="./style_lora_output_flow_optimized"

# Training settings optimized for 16GB
RESOLUTION=768         # Lower resolution
BATCH_SIZE=1
GRADIENT_ACCUM=16
MAX_STEPS=5000
LEARNING_RATE=1e-4

# LoRA settings - smaller to save memory
LORA_RANK=16           # Reduced from 32
LORA_ALPHA=16

# Flow Matching settings
WEIGHTING_SCHEME="none"
GUIDANCE_SCALE=3.5

# Run training with AGGRESSIVE memory optimization
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
    --enable_cpu_offload \
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
echo "  ✓ Lower resolution (768 instead of 1024)"
echo "  ✓ Smaller LoRA rank (16 instead of 32)"
echo "  ✓ Latent caching"
echo "  ✓ Text embedding caching"
echo "  ✓ ALL models offloaded to CPU when not needed"
echo "  ✓ 8-bit Adam optimizer"
echo "  ✓ Gradient checkpointing"
echo "  ✓ Mixed precision (bf16)"
echo ""
echo "Expected memory usage: ~14-16GB"
echo ""
echo "Note: Training will be slower due to CPU offloading"
