"""
Train a style LoRA on top of the merged Flux model with TransparentVAE
This script trains a new LoRA adapter for style transfer while preserving
the transparency capabilities from the merged base model.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import math
from pathlib import Path
from tqdm.auto import tqdm
import json
from datetime import datetime

from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from lib_layerdiffuse.vae import TransparentVAE
from dataset import create_dataloader

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a style LoRA on Flux with TransparentVAE")

    # Model paths
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to merged base model (output from merge_lora.py)"
    )
    parser.add_argument(
        "--trans_vae",
        type=str,
        default="./models/TransparentVAE.pth",
        help="Path to TransparentVAE weights"
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training image resolution (base size)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Explicit training image height (overrides resolution/aspect_ratio_type)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Explicit training image width (overrides resolution/aspect_ratio_type)"
    )
    parser.add_argument(
        "--aspect_ratio_type",
        type=str,
        default="square",
        choices=["square", "portrait", "landscape", "auto"],
        help="Aspect ratio type: square (1:1), portrait (9:16), landscape (16:9), or auto (keep original)"
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images"
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per device"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_train_epochs if set)"
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
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type"
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (scaling factor)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["to_q", "to_k", "to_v", "to_out.0"],
        help="Target modules for LoRA adaptation"
    )

    # Training settings
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--use_offset",
        action="store_true",
        default=True,
        help="Use transparency offset during encoding"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Checkpointing
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/style_lora",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every N steps"
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        nargs="+",
        default=["a beautiful transparent glass bottle", "a crystal clear ice sculpture"],
        help="Prompts for validation images"
    )

    # Logging
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Logging backend (tensorboard, wandb, or all)"
    )

    # Data loading
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading"
    )

    args = parser.parse_args()
    return args


def encode_prompt(pipe, prompt, device, num_images_per_prompt=1):
    """
    Encode text prompt using Flux's dual text encoders

    Returns:
        prompt_embeds: T5 embeddings
        pooled_prompt_embeds: CLIP pooled embeddings
    """
    with torch.no_grad():
        # Encode with both CLIP and T5
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )

    return prompt_embeds, pooled_prompt_embeds, text_ids


def compute_loss(
    model,
    noise_scheduler,
    latents,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    guidance_scale=3.5,
):
    """
    Compute the diffusion loss for flow matching

    Args:
        model: Flux transformer model
        noise_scheduler: Flow matching scheduler
        latents: Encoded latent representations
        prompt_embeds: Text embeddings from T5
        pooled_prompt_embeds: Pooled text embeddings from CLIP
        text_ids: Text token IDs
        guidance_scale: Classifier-free guidance scale

    Returns:
        loss: MSE loss between predicted and target noise
    """
    bsz = latents.shape[0]

    # Sample random timesteps
    # Flux uses a uniform distribution for flow matching
    timesteps = torch.rand(bsz, device=latents.device)
    timesteps = timesteps.view(bsz)

    # Add noise to latents according to flow matching
    noise = torch.randn_like(latents)

    # Flow matching: interpolate between noise and data
    # x_t = t * x_1 + (1 - t) * x_0
    noisy_latents = timesteps.view(-1, 1, 1, 1) * latents + (1 - timesteps.view(-1, 1, 1, 1)) * noise

    # Prepare latent image ids (positional encoding)
    latent_image_ids = noise_scheduler._prepare_latent_image_ids(
        bsz,
        latents.shape[2] // 2,  # Height in patches
        latents.shape[3] // 2,  # Width in patches
        latents.device,
        latents.dtype,
    )

    # Predict the velocity (target is data - noise)
    # For flow matching: v = x_1 - x_0 = latents - noise
    target = latents - noise

    # Pack latents (Flux uses 2x2 patch packing)
    packed_noisy_latents = noise_scheduler._pack_latents(
        noisy_latents,
        bsz,
        latents.shape[2],
        latents.shape[3],
    )

    # Expand timesteps to match the packed format
    # Flux uses 1000 timesteps internally
    timesteps_expanded = timesteps * 1000

    # Forward pass through the model
    model_pred = model(
        hidden_states=packed_noisy_latents,
        timestep=timesteps_expanded,
        guidance=torch.tensor([guidance_scale], device=latents.device, dtype=latents.dtype).expand(bsz),
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]

    # Unpack the predictions
    model_pred = noise_scheduler._unpack_latents(
        model_pred,
        latents.shape[2],
        latents.shape[3],
        noise_scheduler.config.vae_scale_factor,
    )

    # Compute MSE loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def main():
    args = parse_args()

    # Setup accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

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

    # Freeze VAE and text encoders
    pipe.vae.requires_grad_(False)
    trans_vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    # Setup LoRA for transformer
    logger.info("Setting up LoRA adapter")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
    )

    # Apply LoRA to transformer
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Create dataloaders
    logger.info(f"Loading training data from {args.data_dir}")
    train_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
        shuffle=True,
        aspect_ratio_type=args.aspect_ratio_type,
        height=args.height,
        width=args.width,
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        pipe.transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Setup learning rate scheduler
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
    pipe.vae.to(accelerator.device)
    trans_vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)

    # Training info
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(args.num_train_epochs):
        pipe.transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(pipe.transformer):
                # Move batch to device
                img_rgba = batch["img_rgba"].to(accelerator.device, dtype=torch.float32)
                img_rgb = batch["img_rgb"].to(accelerator.device, dtype=pipe.vae.dtype)
                padded_rgb = batch["padded_rgb"].to(accelerator.device, dtype=torch.float32)
                captions = batch["captions"]

                # Encode images to latents using TransparentVAE
                with torch.no_grad():
                    latents = trans_vae.encode(
                        img_rgba=img_rgba,
                        img_rgb=img_rgb,
                        padded_img_rgb=padded_rgb,
                        use_offset=args.use_offset,
                    )

                # Encode text prompts
                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                        pipe, captions[0] if len(captions) == 1 else captions,
                        accelerator.device, num_images_per_prompt=1
                    )

                # Compute loss
                loss = compute_loss(
                    model=pipe.transformer,
                    noise_scheduler=pipe.scheduler,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    guidance_scale=args.guidance_scale,
                )

                # Backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {save_path}")

                        # Save LoRA weights
                        unwrapped_model = accelerator.unwrap_model(pipe.transformer)
                        unwrapped_model.save_pretrained(save_path)

                        # Save training state
                        accelerator.save_state(save_path)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final_model")
        logger.info(f"Saving final model to {save_path}")

        unwrapped_model = accelerator.unwrap_model(pipe.transformer)
        unwrapped_model.save_pretrained(save_path)

        # Save metadata
        metadata = {
            "base_model": args.base_model,
            "trans_vae": args.trans_vae,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "training_steps": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
