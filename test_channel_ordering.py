"""
Test script to determine correct channel ordering for TransparentVAE encoder
This script tests both [padded_rgb, alpha] and [alpha, padded_rgb] orderings
"""

import torch
import argparse
from diffusers import FluxPipeline
from lib_layerdiffuse.vae import TransparentVAE, dist_sample_deterministic
from dataset import create_dataloader
import copy


def test_channel_order(trans_vae, img_rgba, img_rgb, padded_rgb, order="rgb_alpha"):
    """
    Test encoding with different channel orders

    Args:
        order: "rgb_alpha" or "alpha_rgb"
    """
    print(f"\n{'='*80}")
    print(f"Testing channel order: {order}")
    print(f"{'='*80}")

    with torch.no_grad():
        # Extract alpha channel
        a_bchw_01 = img_rgba[:, 3:, :, :]

        # Encode with VAE
        vae_feed = img_rgb.to(device=trans_vae.sd_vae.device, dtype=trans_vae.sd_vae.dtype)
        latent_dist = trans_vae.sd_vae.encode(vae_feed).latent_dist

        # Test different channel orderings for offset_feed
        if order == "rgb_alpha":
            offset_feed = torch.cat([padded_rgb, a_bchw_01], dim=1)
            print(f"  Channel order: [padded_rgb(3ch), alpha(1ch)] = 4 channels")
        else:  # alpha_rgb
            offset_feed = torch.cat([a_bchw_01, padded_rgb], dim=1)
            print(f"  Channel order: [alpha(1ch), padded_rgb(3ch)] = 4 channels")

        offset_feed = offset_feed.to(device=trans_vae.sd_vae.device, dtype=trans_vae.dtype)

        print(f"  offset_feed shape: {offset_feed.shape}")
        print(f"  offset_feed range: [{offset_feed.min():.4f}, {offset_feed.max():.4f}]")
        print(f"  offset_feed mean: {offset_feed.mean():.4f}, std: {offset_feed.std():.4f}")

        # Run through encoder
        encoder_output = trans_vae.encoder(offset_feed)
        print(f"\n  encoder_output:")
        print(f"    shape: {encoder_output.shape}")
        print(f"    range: [{encoder_output.min():.6f}, {encoder_output.max():.6f}]")
        print(f"    mean: {encoder_output.mean():.6f}, std: {encoder_output.std():.6f}")
        print(f"    has_nan: {torch.isnan(encoder_output).any()}")
        print(f"    has_inf: {torch.isinf(encoder_output).any()}")

        if torch.isnan(encoder_output).any():
            print(f"  ❌ Encoder output contains NaN with {order} ordering!")
            return False

        # Multiply by alpha
        offset = encoder_output * trans_vae.alpha
        print(f"\n  offset (after * {trans_vae.alpha}):")
        print(f"    range: [{offset.min():.6f}, {offset.max():.6f}]")
        print(f"    mean: {offset.mean():.6f}, std: {offset.std():.6f}")
        print(f"    has_nan: {torch.isnan(offset).any()}")
        print(f"    has_inf: {torch.isinf(offset).any()}")

        if torch.isnan(offset).any():
            print(f"  ❌ Offset contains NaN after alpha multiplication with {order} ordering!")
            return False

        # Sample with perturbation
        try:
            latent = dist_sample_deterministic(dist=latent_dist, perturbation=offset)
            print(f"\n  latent (after sampling):")
            print(f"    range: [{latent.min():.6f}, {latent.max():.6f}]")
            print(f"    mean: {latent.mean():.6f}, std: {latent.std():.6f}")
            print(f"    has_nan: {torch.isnan(latent).any()}")

            if torch.isnan(latent).any():
                print(f"  ❌ Latent contains NaN with {order} ordering!")
                return False

            # Apply scaling
            latent = trans_vae.sd_vae.config.scaling_factor * (latent - trans_vae.sd_vae.config.shift_factor)
            print(f"\n  latent (final, after scaling):")
            print(f"    range: [{latent.min():.6f}, {latent.max():.6f}]")
            print(f"    mean: {latent.mean():.6f}, std: {latent.std():.6f}")
            print(f"    has_nan: {torch.isnan(latent).any()}")

            if torch.isnan(latent).any():
                print(f"  ❌ Final latent contains NaN with {order} ordering!")
                return False

            print(f"\n  ✓ Encoding successful with {order} ordering!")
            return True

        except Exception as e:
            print(f"  ❌ Error during sampling: {e}")
            return False


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
    trans_vae.to(device)
    pipe.vae.to(device)
    print(f"Models loaded!")

    # Load data
    print("\nCreating dataloader...")
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=1,
        num_workers=0,
        resolution=args.resolution,
        aspect_ratio_type=args.aspect_ratio_type,
        shuffle=False,
    )

    # Get one batch
    batch = next(iter(dataloader))
    img_rgba = batch["img_rgba"].to(device, dtype=torch.float32)
    img_rgb = batch["img_rgb"].to(device, dtype=pipe.vae.dtype)
    padded_rgb = batch["padded_rgb"].to(device, dtype=torch.float32)

    print(f"\nInput data:")
    print(f"  img_rgba: {img_rgba.shape}, range=[{img_rgba.min():.4f}, {img_rgba.max():.4f}]")
    print(f"  img_rgb: {img_rgb.shape}, range=[{img_rgb.min():.4f}, {img_rgb.max():.4f}]")
    print(f"  padded_rgb: {padded_rgb.shape}, range=[{padded_rgb.min():.4f}, {padded_rgb.max():.4f}]")

    # Test both orderings
    print("\n" + "="*80)
    print("TESTING BOTH CHANNEL ORDERINGS")
    print("="*80)

    result_rgb_alpha = test_channel_order(trans_vae, img_rgba, img_rgb, padded_rgb, order="rgb_alpha")
    result_alpha_rgb = test_channel_order(trans_vae, img_rgba, img_rgb, padded_rgb, order="alpha_rgb")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"  [padded_rgb, alpha] ordering: {'✓ SUCCESS' if result_rgb_alpha else '❌ FAILED'}")
    print(f"  [alpha, padded_rgb] ordering: {'✓ SUCCESS' if result_alpha_rgb else '❌ FAILED'}")
    print("="*80)

    if result_rgb_alpha and not result_alpha_rgb:
        print("\n✓ Current implementation is CORRECT: use [padded_rgb, alpha]")
        print("  No changes needed!")
    elif result_alpha_rgb and not result_rgb_alpha:
        print("\n⚠️ WRONG ORDER DETECTED!")
        print("  Current: [padded_rgb, alpha]")
        print("  Should be: [alpha, padded_rgb]")
        print("  Need to fix lib_layerdiffuse/vae.py line 284")
    elif result_rgb_alpha and result_alpha_rgb:
        print("\n⚠️ Both orderings work - encoder might be order-agnostic or untrained")
    else:
        print("\n❌ Both orderings fail - there's a deeper issue")
        print("  Check encoder weights, alpha value, or input data")

    print("="*80)


if __name__ == "__main__":
    main()
