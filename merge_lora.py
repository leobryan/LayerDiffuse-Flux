"""
Merge LoRA weights into base Flux model
This script merges the layerlora.safetensors into the base Flux model weights
to create a new base model for further fine-tuning.
"""

import torch
import argparse
import os
from diffusers import FluxPipeline
from safetensors.torch import save_file
from tqdm import tqdm


def merge_lora_to_base(pipe, lora_scale=1.0):
    """
    Merge LoRA weights into the base transformer weights

    Args:
        pipe: FluxPipeline with loaded LoRA
        lora_scale: Scale factor for LoRA weights (default: 1.0 for full merge)

    Returns:
        Modified pipeline with merged weights
    """
    print(f"Merging LoRA weights with scale {lora_scale}...")

    # Get the transformer model
    transformer = pipe.transformer

    # Fuse LoRA weights into base model
    # This uses diffusers' built-in LoRA fusion method
    pipe.fuse_lora(lora_scale=lora_scale)

    print("LoRA weights successfully merged into base model!")
    return pipe


def save_merged_model(pipe, output_dir):
    """
    Save the merged model to disk

    Args:
        pipe: FluxPipeline with merged weights
        output_dir: Directory to save the merged model
    """
    print(f"Saving merged model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save the entire pipeline
    pipe.save_pretrained(output_dir)

    print(f"Merged model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into Flux base model")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to base Flux model (e.g., black-forest-labs/FLUX.1-dev)"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default="./models/layerlora.safetensors",
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/flux_merged_base",
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale factor for LoRA merge (0.0-1.0, default: 1.0)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype"
    )

    args = parser.parse_args()

    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]

    print("="*80)
    print("Flux LoRA Merge Script")
    print("="*80)
    print(f"Base Model: {args.base_model}")
    print(f"LoRA Weights: {args.lora_weights}")
    print(f"Output Directory: {args.output_dir}")
    print(f"LoRA Scale: {args.lora_scale}")
    print(f"Dtype: {args.dtype}")
    print("="*80)

    # Load base model
    print("\nLoading base Flux model...")
    pipe = FluxPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype
    )
    print("Base model loaded successfully!")

    # Load LoRA weights
    print(f"\nLoading LoRA weights from {args.lora_weights}...")
    pipe.load_lora_weights(args.lora_weights)
    print("LoRA weights loaded successfully!")

    # Merge LoRA into base model
    pipe = merge_lora_to_base(pipe, lora_scale=args.lora_scale)

    # Unload LoRA adapters after merging
    pipe.unload_lora_weights()

    # Save merged model
    save_merged_model(pipe, args.output_dir)

    print("\n" + "="*80)
    print("Merge completed successfully!")
    print(f"Merged model saved to: {args.output_dir}")
    print("="*80)
    print("\nYou can now use this merged model as the base for style fine-tuning:")
    print(f"  python train_style_lora.py --base_model {args.output_dir}")


if __name__ == "__main__":
    main()
