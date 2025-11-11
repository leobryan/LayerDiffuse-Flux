#!/bin/bash
# Example training configuration script for style LoRA fine-tuning
# Copy this file and modify the parameters as needed

# ============================================================================
# Step 1: Merge the original layerlora into base model
# ============================================================================
echo "Step 1: Merging layerlora into base Flux model..."

python merge_lora.py \
    --base_model "black-forest-labs/FLUX.1-dev" \
    --lora_weights "./models/layerlora.safetensors" \
    --output_dir "./models/flux_merged_base" \
    --lora_scale 1.0 \
    --dtype "bfloat16"

echo "Merge completed! Merged model saved to ./models/flux_merged_base"

# ============================================================================
# Step 2: Train a new style LoRA on the merged model
# ============================================================================
echo ""
echo "Step 2: Training style LoRA..."

# Accelerate configuration for single GPU
export CUDA_VISIBLE_DEVICES=0

accelerate launch --mixed_precision="bf16" train_style_lora.py \
    --base_model "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --data_dir "./training_data" \
    --output_dir "./output/style_lora_$(date +%Y%m%d_%H%M%S)" \
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
        "a beautiful transparent glass bottle in anime style" \
        "a crystal sculpture in watercolor style" \
        "a transparent butterfly in oil painting style" \
    --logging_dir "logs" \
    --report_to "tensorboard" \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --seed 42

echo ""
echo "Training completed! Check the output directory for trained LoRA weights."

# ============================================================================
# Step 3: Test the trained style LoRA
# ============================================================================
echo ""
echo "Step 3: Testing the trained style LoRA..."

# Update this path to your trained LoRA checkpoint
TRAINED_LORA_PATH="./output/style_lora_XXXXXX/final_model"

python demo_t2i.py \
    --ckpt_path "./models/flux_merged_base" \
    --trans_vae "./models/TransparentVAE.pth" \
    --lora_weights "$TRAINED_LORA_PATH" \
    --output_dir "./test_outputs" \
    --prompt "a beautiful transparent glass bottle in my custom style" \
    --steps 50 \
    --guidance 3.5 \
    --seed 12345

echo ""
echo "Test image generated! Check ./test_outputs/"
