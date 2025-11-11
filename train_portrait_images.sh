#!/bin/bash
# Training configuration for 9:16 portrait images (e.g., mobile screen content)
# This script is optimized for training with portrait-oriented transparent images

echo "=================================="
echo "Training Style LoRA for 9:16 Portrait Images"
echo "=================================="

# ============================================================================
# Step 1: Merge the original layerlora into base model (if not already done)
# ============================================================================
if [ ! -d "./models/flux_merged_base" ]; then
    echo "Merged base model not found. Running merge first..."
    python merge_lora.py \
        --base_model "black-forest-labs/FLUX.1-dev" \
        --lora_weights "./models/layerlora.safetensors" \
        --output_dir "./models/flux_merged_base" \
        --lora_scale 1.0 \
        --dtype "bfloat16"
    echo "Merge completed!"
else
    echo "Merged base model found, skipping merge step."
fi

# ============================================================================
# Step 2: Train style LoRA for portrait images
# ============================================================================
echo ""
echo "Starting training for portrait images..."

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Option 1: Use preset portrait aspect ratio (768x1360)
# Standard mode (requires ~24GB VRAM)
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_portrait_$(date +%Y%m%d_%H%M%S)" \
    --aspect_ratio_type "portrait" \
    --resolution 1024 \
    --center_crop \
    --random_flip \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 100 \
    --learning_rate 1e-4 \
    --lr_scheduler "constant_with_warmup" \
    --lr_warmup_steps 500 \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --lora_target_modules "to_q" "to_k" "to_v" "to_out.0" \
    --guidance_scale 3.5 \
    --use_offset \
    --mixed_precision "bf16" \
    --gradient_checkpointing \
    --checkpointing_steps 500 \
    --validation_steps 500 \
    --validation_prompts \
        "a beautiful transparent portrait in anime style" \
        "a crystal figure on mobile screen" \
        "a transparent character art in portrait format" \
    --logging_dir "logs" \
    --report_to "tensorboard" \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --seed 42

# If you encounter OOM (Out of Memory) errors, use CPU offload mode:
# (Slower but uses only ~16GB VRAM)
# Add --enable_cpu_offload flag to the above command

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo ""
echo "Alternative configurations:"
echo ""
echo "For custom portrait size (576x1024, uses less VRAM):"
echo "  --height 1024 --width 576"
echo ""
echo "For higher quality portrait (1088x1920):"
echo "  --height 1920 --width 1088"
