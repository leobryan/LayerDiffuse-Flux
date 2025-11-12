"""
Inference script for full fine-tuned Flux model with TransparentVAE
Use this when you have a full transformer model (not LoRA)
"""

import torch
import argparse
import os
import datetime
from diffusers import FluxPipeline, FluxTransformer2DModel
from lib_layerdiffuse.vae import TransparentVAE
from PIL import Image
import numpy as np


def generate_img(pipe, trans_vae, args):
    """Generate transparent image using fine-tuned full model"""

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
    parser = argparse.ArgumentParser(description="Generate transparent images with fine-tuned Flux model")

    # Model paths
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base Flux model (can be the merged base model)")
    parser.add_argument("--finetuned_transformer", type=str, required=True,
                        help="Path to fine-tuned transformer checkpoint directory")
    parser.add_argument("--trans_vae", type=str, default="./models/TransparentVAE.pth",
                        help="Path to TransparentVAE weights")

    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str, default="./inference_outputs",
                        help="Directory to save generated images")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (without extension). If not specified, uses timestamp")

    # Image parameters
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width (must be divisible by 128)")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height (must be divisible by 128)")

    # Generation settings
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=3.5,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Model settings
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")

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

    # Load fine-tuned transformer
    print(f"\n2. Loading fine-tuned transformer from: {args.finetuned_transformer}")
    finetuned_transformer = FluxTransformer2DModel.from_pretrained(
        args.finetuned_transformer,
        torch_dtype=dtype,
    )

    # Replace transformer in pipeline
    pipe.transformer = finetuned_transformer
    pipe = pipe.to('cuda')

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
