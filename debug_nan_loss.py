"""
Debug script to diagnose NaN loss issues
Run this before training to check data and model integrity
"""

import torch
import argparse
from diffusers import FluxPipeline
from lib_layerdiffuse.vae import TransparentVAE
from dataset import create_dataloader
import numpy as np


def check_data(dataloader, num_batches=5):
    """Check if training data contains NaN or Inf"""
    print("=" * 80)
    print("Checking training data...")
    print("=" * 80)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        img_rgba = batch["img_rgba"]
        img_rgb = batch["img_rgb"]
        padded_rgb = batch["padded_rgb"]

        print(f"\nBatch {i}:")
        print(f"  img_rgba: shape={img_rgba.shape}, "
              f"min={img_rgba.min():.4f}, max={img_rgba.max():.4f}, "
              f"has_nan={torch.isnan(img_rgba).any()}, has_inf={torch.isinf(img_rgba).any()}")
        print(f"  img_rgb: shape={img_rgb.shape}, "
              f"min={img_rgb.min():.4f}, max={img_rgb.max():.4f}, "
              f"has_nan={torch.isnan(img_rgb).any()}, has_inf={torch.isinf(img_rgb).any()}")
        print(f"  padded_rgb: shape={padded_rgb.shape}, "
              f"min={padded_rgb.min():.4f}, max={padded_rgb.max():.4f}, "
              f"has_nan={torch.isnan(padded_rgb).any()}, has_inf={torch.isinf(padded_rgb).any()}")

        # Check for extreme values
        if torch.abs(img_rgb).max() > 100:
            print(f"  ⚠️ WARNING: img_rgb has extreme values!")
        if torch.abs(padded_rgb).max() > 10:
            print(f"  ⚠️ WARNING: padded_rgb has extreme values!")

    print("\n✓ Data check completed")


def check_vae_encoding(pipe, trans_vae, dataloader, device):
    """Check if VAE encoding produces valid latents"""
    print("\n" + "=" * 80)
    print("Checking VAE encoding...")
    print("=" * 80)

    trans_vae.to(device)
    pipe.vae.to(device)

    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        img_rgba = batch["img_rgba"].to(device, dtype=torch.float32)
        img_rgb = batch["img_rgb"].to(device, dtype=pipe.vae.dtype)
        padded_rgb = batch["padded_rgb"].to(device, dtype=torch.float32)

        print(f"\nBatch {i}:")

        with torch.no_grad():
            try:
                latents = trans_vae.encode(
                    img_rgba=img_rgba,
                    img_rgb=img_rgb,
                    padded_img_rgb=padded_rgb,
                    use_offset=True,
                )

                print(f"  Latents: shape={latents.shape}, "
                      f"min={latents.min():.4f}, max={latents.max():.4f}, "
                      f"mean={latents.mean():.4f}, std={latents.std():.4f}")
                print(f"  has_nan={torch.isnan(latents).any()}, "
                      f"has_inf={torch.isinf(latents).any()}")

                if torch.isnan(latents).any():
                    print(f"  ❌ ERROR: Latents contain NaN!")
                    return False
                if torch.isinf(latents).any():
                    print(f"  ❌ ERROR: Latents contain Inf!")
                    return False
                if torch.abs(latents).max() > 1000:
                    print(f"  ⚠️ WARNING: Latents have extreme values (max: {latents.abs().max():.2f})")

            except Exception as e:
                print(f"  ❌ ERROR during encoding: {e}")
                return False

    print("\n✓ VAE encoding check completed")
    return True


def check_model_weights(pipe):
    """Check if model weights contain NaN or Inf"""
    print("\n" + "=" * 80)
    print("Checking model weights...")
    print("=" * 80)

    has_nan = False
    has_inf = False

    for name, param in pipe.transformer.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ Parameter {name} contains NaN!")
            has_nan = True
        if torch.isinf(param).any():
            print(f"  ❌ Parameter {name} contains Inf!")
            has_inf = True

    if not has_nan and not has_inf:
        print("  ✓ All model weights are valid")
        return True
    return False


def test_forward_pass(pipe, trans_vae, dataloader, device):
    """Test a single forward pass"""
    print("\n" + "=" * 80)
    print("Testing forward pass...")
    print("=" * 80)

    trans_vae.to(device)
    pipe.vae.to(device)
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    pipe.transformer.to(device)

    batch = next(iter(dataloader))

    img_rgba = batch["img_rgba"].to(device, dtype=torch.float32)
    img_rgb = batch["img_rgb"].to(device, dtype=pipe.vae.dtype)
    padded_rgb = batch["padded_rgb"].to(device, dtype=torch.float32)
    caption = batch["captions"][0]

    print(f"Caption: {caption}")

    with torch.no_grad():
        # Encode image
        print("\n1. Encoding image...")
        latents = trans_vae.encode(
            img_rgba=img_rgba,
            img_rgb=img_rgb,
            padded_img_rgb=padded_rgb,
            use_offset=True,
        )
        print(f"   Latents: {latents.shape}, range=[{latents.min():.4f}, {latents.max():.4f}]")

        # Encode prompt
        print("\n2. Encoding prompt...")
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=caption,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
        )
        print(f"   Prompt embeds: {prompt_embeds.shape}")
        print(f"   Pooled embeds: {pooled_prompt_embeds.shape}")

        # Sample timesteps
        print("\n3. Adding noise...")
        bsz = latents.shape[0]
        timesteps = torch.rand(bsz, device=device)
        noise = torch.randn_like(latents)
        noisy_latents = timesteps.view(-1, 1, 1, 1) * latents + (1 - timesteps.view(-1, 1, 1, 1)) * noise

        print(f"   Timesteps: {timesteps}")
        print(f"   Noisy latents: range=[{noisy_latents.min():.4f}, {noisy_latents.max():.4f}]")

        # Check for NaN after noise addition
        if torch.isnan(noisy_latents).any():
            print("   ❌ NaN detected after adding noise!")
            print(f"      - latents has NaN: {torch.isnan(latents).any()}")
            print(f"      - noise has NaN: {torch.isnan(noise).any()}")
            return False

        print("\n✓ Forward pass check completed")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--trans_vae", type=str, default="./models/TransparentVAE.pth")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--aspect_ratio_type", type=str, default="square")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models
    print("\nLoading models...")
    pipe = FluxPipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)
    trans_vae.load_state_dict(torch.load(args.trans_vae), strict=False)
    print("Models loaded!")

    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0,
        resolution=args.resolution,
        aspect_ratio_type=args.aspect_ratio_type,
        shuffle=False,
    )
    print(f"Dataloader created with {len(dataloader.dataset)} samples")

    # Run checks
    print("\n" + "=" * 80)
    print("STARTING DIAGNOSTICS")
    print("=" * 80)

    # 1. Check data
    check_data(dataloader, num_batches=5)

    # 2. Check VAE encoding
    vae_ok = check_vae_encoding(pipe, trans_vae, dataloader, device)

    # 3. Check model weights
    weights_ok = check_model_weights(pipe)

    # 4. Test forward pass
    if vae_ok and weights_ok:
        forward_ok = test_forward_pass(pipe, trans_vae, dataloader, device)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"  VAE Encoding: {'✓ PASS' if vae_ok else '❌ FAIL'}")
    print(f"  Model Weights: {'✓ PASS' if weights_ok else '❌ FAIL'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
