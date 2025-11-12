"""
Load and use LoRA weights saved in older format (model.safetensors / adapter_model.safetensors)
This script handles PEFT-style LoRA weights that were saved with save_pretrained()
"""

import torch
import argparse
import os
import datetime
from diffusers import FluxPipeline
from lib_layerdiffuse.vae import TransparentVAE
from PIL import Image
import numpy as np
from safetensors.torch import load_file
from peft import PeftModel, LoraConfig


def load_old_lora_weights(pipe, lora_path):
    """
    Load LoRA weights from old format (PEFT style)

    Args:
        pipe: FluxPipeline instance
        lora_path: Path to checkpoint directory containing adapter_model.safetensors or model.safetensors
    """
    print(f"\nLoading LoRA weights from: {lora_path}")

    # Check what files exist
    adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
    model_path = os.path.join(lora_path, "model.safetensors")

    if os.path.exists(adapter_path):
        print(f"Found: adapter_model.safetensors")
        lora_file = adapter_path
    elif os.path.exists(model_path):
        print(f"Found: model.safetensors")
        lora_file = model_path
    else:
        raise FileNotFoundError(f"No LoRA weights found in {lora_path}")

    # Load the state dict
    state_dict = load_file(lora_file)
    print(f"Loaded {len(state_dict)} keys from {os.path.basename(lora_file)}")

    # Try to load as PEFT model
    try:
        print("\nAttempting to load as PEFT LoRA...")
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            lora_path,
            is_trainable=False
        )
        print("✓ Successfully loaded as PEFT LoRA")
        return True
    except Exception as e:
        print(f"Failed to load as PEFT: {e}")

    # If PEFT loading failed, try manual loading
    print("\nAttempting manual LoRA weight injection...")
    try:
        # Load weights directly into transformer
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")

        if len(missing) < len(state_dict):
            print("✓ Partial success - some weights loaded")
            return True
        else:
            print("❌ Failed to load weights")
            return False

    except Exception as e:
        print(f"Manual loading failed: {e}")
        return False


def generate_img(pipe, trans_vae, args):
    """Generate transparent image"""

    print(f"\nGenerating image...")
    print(f"Prompt: {args.prompt}")
    print(f"Size: {args.width}x{args.height}")
    print(f"Steps: {args.steps}")
    print(f"Guidance: {args.guidance}")
    print(f"Seed: {args.seed}")

    # Generate latents
    latents = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        output_type="latent",
        generator=torch.Generator("cuda").manual_seed(args.seed),
        guidance_scale=args.guidance,
    ).images

    # Unpack latents
    latents = pipe._unpack_latents(latents, args.height, args.width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    # Decode with TransparentVAE
    print("Decoding with TransparentVAE...")
    with torch.no_grad():
        original_x, x = trans_vae.decode(latents)

    # Convert to image
    x = x.clamp(0, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((x*255).float().cpu().numpy().astype(np.uint8)[0])

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with old format LoRA weights")

    # Model paths
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base Flux model (merged base model)")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                        help="Path to checkpoint directory containing adapter_model.safetensors or model.safetensors")
    parser.add_argument("--trans_vae", type=str, default="./models/TransparentVAE.pth",
                        help="Path to TransparentVAE weights")

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Directory to save generated images")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (without extension)")

    # Image parameters
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    # Generation settings
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=42)

    # Model settings
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])

    args = parser.parse_args()

    # Validate dimensions
    if args.width % 128 != 0 or args.height % 128 != 0:
        raise ValueError(f"Width and height must be divisible by 128. Got {args.width}x{args.height}")

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("="*80)
    print("Loading models...")
    print("="*80)

    # Load base pipeline
    print(f"\n1. Loading base Flux pipeline from: {args.base_model}")
    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe = pipe.to('cuda')

    # Load LoRA weights
    print(f"\n2. Loading LoRA weights...")
    success = load_old_lora_weights(pipe, args.lora_checkpoint)

    if not success:
        print("\n" + "="*80)
        print("❌ Failed to load LoRA weights!")
        print("="*80)
        print("\nPlease check:")
        print(f"1. Checkpoint path: {args.lora_checkpoint}")
        print("2. Files in checkpoint:")
        for f in os.listdir(args.lora_checkpoint):
            print(f"   - {f}")
        exit(1)

    # Load TransparentVAE
    print(f"\n3. Loading TransparentVAE from: {args.trans_vae}")
    trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)
    trans_vae.load_state_dict(torch.load(args.trans_vae), strict=False)
    trans_vae.to('cuda')

    print("\n" + "="*80)
    print("✓ All models loaded successfully!")
    print("="*80)

    # Generate image
    img = generate_img(pipe, trans_vae, args)

    # Save image
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_name:
        output_filename = f"{args.output_name}.png"
    else:
        output_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    output_path = os.path.join(args.output_dir, output_filename)
    img.save(output_path)

    print("\n" + "="*80)
    print(f"✓ Image saved to: {output_path}")
    print(f"  Size: {img.size}")
    print("="*80)
