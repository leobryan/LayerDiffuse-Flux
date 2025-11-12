"""
Style LoRA training with Flow Matching loss (latent-space)
Based on Hugging Face's official Flux DreamBooth training script
Uses velocity prediction and timestep weighting schemes
"""

import argparse
import logging
import math
import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from lib_layerdiffuse.vae import TransparentVAE
from dataset import create_dataloader


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train style LoRA with Flow Matching loss")

    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to merged base model"
    )
    parser.add_argument(
        "--trans_vae",
        type=str,
        default="./models/TransparentVAE.pth",
        help="Path to TransparentVAE weights"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Training data directory"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution"
    )
    parser.add_argument(
        "--aspect_ratio_type",
        type=str,
        default="square",
        choices=["square", "portrait", "landscape", "auto"],
        help="Aspect ratio type"
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Center crop images"
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Random horizontal flip"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./style_lora_output_flow",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant"],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA (if None, uses default Flux attention modules)"
    )

    # Flow Matching arguments
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Weighting scheme for timestep sampling"
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean for logit_normal weighting"
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="Std for logit_normal weighting"
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale for mode weighting"
    )

    # Generation arguments
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--disable_offset",
        action="store_true",
        help="Disable transparency offset during encoding"
    )

    # Optimization arguments
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable CPU offloading for VAE"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )

    args = parser.parse_args()
    return args


def compute_flow_matching_loss(
    model_pred,
    target,
    weighting,
):
    """
    Compute Flow Matching loss with weighting

    Args:
        model_pred: Model prediction (B, C, H, W)
        target: Target (noise - model_input) (B, C, H, W)
        weighting: Loss weighting (B, 1, 1, 1)

    Returns:
        loss: Scalar loss
    """
    # Compute weighted MSE loss
    # Shape: (B, C, H, W) -> (B, -1)
    loss = torch.mean(
        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        dim=1,
    )
    loss = loss.mean()

    return loss


def get_sigmas(timesteps, noise_scheduler, n_dim=4, dtype=torch.float32):
    """Get sigmas for given timesteps"""
    sigmas = noise_scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(timesteps.device)

    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Load models
    logger.info(f"Loading base model from {args.base_model}")
    pipe = FluxPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    # Load TransparentVAE
    logger.info(f"Loading TransparentVAE from {args.trans_vae}")
    trans_vae = TransparentVAE(pipe.vae, pipe.vae.dtype)
    trans_vae.load_state_dict(torch.load(args.trans_vae), strict=False)

    # Log settings
    use_transparency_encoder = not args.disable_offset
    logger.info(f"Transparency offset encoder: {'ENABLED' if use_transparency_encoder else 'DISABLED'}")
    logger.info(f"Training with FLOW MATCHING LOSS")
    logger.info(f"  - Weighting scheme: {args.weighting_scheme}")
    logger.info(f"  - Target: velocity (noise - model_input)")

    # Freeze models
    pipe.vae.requires_grad_(False)
    trans_vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    # Set to eval mode
    pipe.vae.eval()
    trans_vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    # Setup LoRA
    logger.info("Setting up LoRA adapter")

    # Default Flux attention modules
    if args.lora_target_modules is None:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
        ]
    else:
        target_modules = args.lora_target_modules

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
    )
    pipe.transformer.add_adapter(lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.transformer.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    # Create noise scheduler for training
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model,
        subfolder="scheduler"
    )

    # Create dataset
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.train_batch_size,
        resolution=args.resolution,
        aspect_ratio_type=args.aspect_ratio_type,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        shuffle=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Learning rate scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    pipe.transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        pipe.transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Move other components to device
    if not args.enable_cpu_offload:
        pipe.vae.to(accelerator.device)
        trans_vae.to(accelerator.device)
        pipe.text_encoder.to(accelerator.device)
        pipe.text_encoder_2.to(accelerator.device)

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        pipe.transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipe.transformer):
                # Move batch to device
                img_rgba = batch["img_rgba"].to(accelerator.device, dtype=torch.float32)
                img_rgb = batch["img_rgb"].to(accelerator.device, dtype=pipe.vae.dtype)
                padded_rgb = batch["padded_rgb"].to(accelerator.device, dtype=torch.float32)
                captions = batch["captions"]

                batch_size = img_rgba.shape[0]
                height, width = img_rgba.shape[2], img_rgba.shape[3]

                # Encode to latents
                with torch.no_grad():
                    if args.enable_cpu_offload:
                        pipe.vae.to(accelerator.device)
                        trans_vae.to(accelerator.device)

                    # Encode with TransparentVAE
                    model_input = trans_vae.encode(
                        img_rgba=img_rgba,
                        img_rgb=img_rgb,
                        padded_img_rgb=padded_rgb,
                        use_offset=use_transparency_encoder,
                    )

                    # Encode prompts
                    if args.enable_cpu_offload:
                        pipe.text_encoder.to(accelerator.device)
                        pipe.text_encoder_2.to(accelerator.device)

                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=captions,
                        prompt_2=None,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                    )

                    if args.enable_cpu_offload:
                        pipe.vae.to("cpu")
                        trans_vae.to("cpu")
                        pipe.text_encoder.to("cpu")
                        pipe.text_encoder_2.to("cpu")
                        torch.cuda.empty_cache()

                # Convert to weight dtype
                model_input = model_input.to(dtype=weight_dtype)

                # Prepare latent image ids (Flux positional encoding)
                latent_image_ids = pipe._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # Sample noise
                noise = torch.randn_like(model_input)

                # Sample timesteps with weighting scheme
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching
                # zt = (1 - sigma) * x + sigma * noise
                sigmas = get_sigmas(timesteps, noise_scheduler, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Pack latents for Flux
                packed_noisy_model_input = pipe._pack_latents(
                    noisy_model_input,
                    batch_size=batch_size,
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # Prepare guidance
                guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                guidance = guidance.expand(batch_size)

                # Forward pass
                model_pred = pipe.transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # Unpack latents
                vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
                model_pred = pipe._unpack_latents(
                    model_pred,
                    height=height,
                    width=width,
                    vae_scale_factor=vae_scale_factor,
                )

                # Compute loss weighting
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas
                )

                # Flow matching target: velocity (noise - model_input)
                target = noise - model_input

                # Compute loss
                loss = compute_flow_matching_loss(
                    model_pred=model_pred,
                    target=target,
                    weighting=weighting,
                )

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Free memory
                del img_rgba, img_rgb, padded_rgb, model_input, noise, model_pred, target
                if args.enable_cpu_offload:
                    torch.cuda.empty_cache()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {save_path}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save LoRA weights
                        unwrapped_transformer = accelerator.unwrap_model(pipe.transformer)
                        pipe.transformer = unwrapped_transformer
                        pipe.save_lora_weights(
                            save_path,
                            safe_serialization=True
                        )
                        pipe.transformer = accelerator.prepare(unwrapped_transformer)

                        # Save training state
                        accelerator.save_state(save_path)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        logger.info(f"Saving final model to {save_path}")
        os.makedirs(save_path, exist_ok=True)

        unwrapped_transformer = accelerator.unwrap_model(pipe.transformer)
        pipe.transformer = unwrapped_transformer
        pipe.save_lora_weights(
            save_path,
            safe_serialization=True
        )

        # Save metadata
        metadata = {
            "base_model": args.base_model,
            "trans_vae": args.trans_vae,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "training_mode": "flow_matching_loss",
            "weighting_scheme": args.weighting_scheme,
            "target": "velocity (noise - model_input)",
            "training_steps": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LoRA weights saved to {save_path}")

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
