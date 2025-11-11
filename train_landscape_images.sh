#!/bin/bash
# Training configuration for 16:9 landscape images (e.g., video content, desktop wallpapers)
# This script is optimized for training with landscape-oriented transparent images

echo "=================================="
echo "Training Style LoRA for 16:9 Landscape Images"
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
# Step 2: Train style LoRA for landscape images
# ============================================================================
echo ""
echo "Starting training for landscape images..."

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Option 1: Use preset landscape aspect ratio (1024x576)
accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_landscape_$(date +%Y%m%d_%H%M%S)" \
    --aspect_ratio_type "landscape" \
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
    --checkpointing_steps 500 \
    --validation_steps 500 \
    --validation_prompts \
        "a beautiful transparent landscape scene in anime style" \
        "a crystal panorama in cinematic style" \
        "a transparent widescreen composition" \
    --logging_dir "logs" \
    --report_to "tensorboard" \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --seed 42

echo ""
echo "=================================="
echo "Training completed!"
echo "=================================="
echo ""
echo "Alternative configurations:"
echo ""
echo "⚠️ IMPORTANT: All dimensions must be multiples of 128!"
echo ""
echo "For custom landscape sizes:"
echo "  --height 512 --width 768   # Lower quality, ~18GB"
echo "  --height 640 --width 1024  # Medium quality, ~25GB"
echo ""
echo "For higher quality landscape:"
echo "  --height 768 --width 1280  # High quality, ~35GB"
echo "  --height 896 --width 1536  # Very high quality, ~40GB+"
echo ""
echo "Note: Larger resolutions require more VRAM!"
