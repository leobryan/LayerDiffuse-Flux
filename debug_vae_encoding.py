"""
Enhanced debug script to pinpoint exact source of NaN in TransparentVAE encoding
This script adds step-by-step validation inside the encoding process
"""

import torch
import argparse
from diffusers import FluxPipeline
from lib_layerdiffuse.vae import TransparentVAE, dist_sample_deterministic
from dataset import create_dataloader
import numpy as np


def check_tensor(name, tensor, step_num=None):
    """Check a tensor for NaN, Inf, and extreme values"""
    prefix = f"  Step {step_num}: " if step_num is not None else "  "

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item() if not has_nan else float('nan')
    max_val = tensor.max().item() if not has_nan else float('nan')
    mean_val = tensor.mean().item() if not has_nan else float('nan')
    std_val = tensor.std().item() if not has_nan else float('nan')

    print(f"{prefix}{name}:")
    print(f"{prefix}  shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"{prefix}  min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
    print(f"{prefix}  has_nan={has_nan}, has_inf={has_inf}")

    if has_nan:
        print(f"{prefix}  ❌ NaN DETECTED!")
        nan_count = torch.isnan(tensor).sum().item()
        total = tensor.numel()
        print(f"{prefix}  NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        return True

    if has_inf:
        print(f"{prefix}  ⚠️ Inf detected!")
        return True

    if torch.abs(tensor).max() > 1e4:
        print(f"{prefix}  ⚠️ Extreme values detected (max abs: {torch.abs(tensor).max():.2e})")
        return True

    return False


def debug_vae_encoding(pipe, trans_vae, dataloader, device):
    """Debug VAE encoding with step-by-step validation"""
    print("=" * 80)
    print("DETAILED VAE ENCODING DEBUG")
    print("=" * 80)

    trans_vae.to(device)
    pipe.vae.to(device)

    # Get one batch
    batch = next(iter(dataloader))

    img_rgba = batch["img_rgba"].to(device, dtype=torch.float32)
    img_rgb = batch["img_rgb"].to(device, dtype=pipe.vae.dtype)
    padded_rgb = batch["padded_rgb"].to(device, dtype=torch.float32)

    print("\n" + "=" * 80)
    print("INPUT VALIDATION")
    print("=" * 80)

    # Check inputs
    print("\nChecking inputs:")
    has_error = False
    has_error |= check_tensor("img_rgba", img_rgba, step_num=0)
    has_error |= check_tensor("img_rgb", img_rgb, step_num=0)
    has_error |= check_tensor("padded_rgb", padded_rgb, step_num=0)

    if has_error:
        print("\n❌ ERROR: Input data contains NaN/Inf!")
        return False

    print("\n✓ All inputs are valid")

    print("\n" + "=" * 80)
    print("ENCODING STEP-BY-STEP")
    print("=" * 80)

    with torch.no_grad():
        # Step 1: Extract alpha channel
        print("\n[Step 1] Extract alpha channel")
        a_bchw_01 = img_rgba[:, 3:, :, :]
        has_error = check_tensor("a_bchw_01 (alpha channel)", a_bchw_01, step_num=1)

        # Step 2: Prepare VAE feed (img_rgb already in [-1, 1])
        print("\n[Step 2] Prepare VAE feed")
        vae_feed = img_rgb.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
        has_error |= check_tensor("vae_feed", vae_feed, step_num=2)

        if has_error:
            print("\n❌ ERROR: NaN detected before VAE encoding!")
            return False

        # Step 3: VAE encode
        print("\n[Step 3] VAE encode")
        try:
            latent_dist = pipe.vae.encode(vae_feed).latent_dist
            print("  VAE encoding successful")

            # Check distribution parameters
            has_error = check_tensor("latent_dist.mean", latent_dist.mean, step_num=3)
            has_error |= check_tensor("latent_dist.std", latent_dist.std, step_num=3)

            if has_error:
                print("\n❌ ERROR: VAE encoding produced NaN in distribution parameters!")
                return False

        except Exception as e:
            print(f"  ❌ ERROR during VAE encoding: {e}")
            return False

        # Step 4: Prepare offset feed
        print("\n[Step 4] Prepare offset feed")
        offset_feed = torch.cat([padded_rgb, a_bchw_01], dim=1).to(device=pipe.vae.device, dtype=trans_vae.dtype)
        has_error = check_tensor("offset_feed (padded_rgb + alpha)", offset_feed, step_num=4)

        if has_error:
            print("\n❌ ERROR: NaN in offset feed!")
            return False

        # Step 5: Encoder forward pass
        print("\n[Step 5] TransparentVAE encoder forward pass")
        try:
            encoder_output = trans_vae.encoder(offset_feed)
            has_error = check_tensor("encoder_output (before * alpha)", encoder_output, step_num=5)

            if has_error:
                print("\n❌ ERROR: Encoder produced NaN!")
                print(f"  Encoder alpha value: {trans_vae.alpha}")
                return False

        except Exception as e:
            print(f"  ❌ ERROR during encoder forward: {e}")
            return False

        # Step 6: Multiply by alpha
        print(f"\n[Step 6] Multiply encoder output by alpha ({trans_vae.alpha})")
        offset = encoder_output * trans_vae.alpha
        has_error = check_tensor("offset (encoder_output * alpha)", offset, step_num=6)

        if has_error:
            print(f"\n❌ ERROR: NaN after multiplying by alpha={trans_vae.alpha}!")
            print("  This suggests alpha value is too large, causing overflow")

            # Try with smaller alpha
            print("\n  Testing with alpha=1.0 (no scaling):")
            test_offset = encoder_output * 1.0
            check_tensor("offset with alpha=1.0", test_offset, step_num=6)

            print("\n  Testing with alpha=10.0:")
            test_offset = encoder_output * 10.0
            check_tensor("offset with alpha=10.0", test_offset, step_num=6)

            return False

        # Step 7: Sample with perturbation
        print("\n[Step 7] Sample latent with perturbation")
        print("  Formula: latent = mean + std * offset")

        # Check intermediate calculation
        std_times_offset = latent_dist.std * offset.to(latent_dist.std)
        has_error = check_tensor("std * offset", std_times_offset, step_num=7)

        if has_error:
            print("\n❌ ERROR: NaN in (std * offset)!")
            return False

        try:
            latent = dist_sample_deterministic(dist=latent_dist, perturbation=offset)
            has_error = check_tensor("latent (after dist_sample_deterministic)", latent, step_num=7)

            if has_error:
                print("\n❌ ERROR: NaN after sampling with perturbation!")
                return False

        except Exception as e:
            print(f"  ❌ ERROR during dist_sample_deterministic: {e}")
            return False

        # Step 8: Apply scaling and shift
        print(f"\n[Step 8] Apply VAE scaling (factor={pipe.vae.config.scaling_factor}, shift={pipe.vae.config.shift_factor})")
        latent = pipe.vae.config.scaling_factor * (latent - pipe.vae.config.shift_factor)
        has_error = check_tensor("latent (final, after scaling)", latent, step_num=8)

        if has_error:
            print(f"\n❌ ERROR: NaN after applying scaling_factor={pipe.vae.config.scaling_factor} and shift_factor={pipe.vae.config.shift_factor}!")
            return False

        print("\n" + "=" * 80)
        print("✓ ALL STEPS PASSED - NO NaN DETECTED!")
        print("=" * 80)
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
    print(f"Models loaded!")
    print(f"TransparentVAE alpha: {trans_vae.alpha}")
    print(f"VAE dtype: {pipe.vae.dtype}")
    print(f"TransparentVAE encoder dtype: {trans_vae.dtype}")

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

    # Run detailed debug
    success = debug_vae_encoding(pipe, trans_vae, dataloader, device)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if success:
        print("✓ VAE encoding works correctly - no NaN detected")
    else:
        print("❌ VAE encoding has issues - see details above")
    print("=" * 80)


if __name__ == "__main__":
    main()
