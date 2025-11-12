"""
Style LoRA training with image-space loss (pixel-space reconstruction)
Loss is computed in image space after decoding, not in latent space
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
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from lib_layerdiffuse.vae import TransparentVAE
from dataset import create_dataloader


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train style LoRA with image-space loss")

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
        default="./style_lora_output_pixel",
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
        default=1,
        help="Batch size (image-space loss needs more memory)"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
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
        default=["to_q", "to_k", "to_v", "to_out.0"],
    )

    # Loss arguments
    parser.add_argument(
        "--rgb_loss_weight",
        type=float,
        default=1.0,
        help="Weight for RGB loss"
    )
    parser.add_argument(
        "--alpha_loss_weight",
        type=float,
        default=1.0,
        help="Weight for alpha channel loss"
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber"],
        help="Loss function type"
    )
    parser.add_argument(
        "--perceptual_loss_weight",
        type=float,
        default=0.0,
        help="Weight for perceptual loss (0 to disable)"
    )

    # Generation arguments
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps (fewer steps for faster training)"
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
        default=["glass bottle, high quality"],
        help="Prompts for validation"
    )

    args = parser.parse_args()
    return args


def compute_image_loss(pred_img, target_img, loss_type="l2", rgb_weight=1.0, alpha_weight=1.0):
    """
    Compute loss in image space

    Args:
        pred_img: (B, 4, H, W) in [0, 1] range (RGB + Alpha)
        target_img: (B, 4, H, W) in [0, 1] range (RGB + Alpha)
        loss_type: "l1", "l2", or "huber"
        rgb_weight: Weight for RGB channels
        alpha_weight: Weight for alpha channel

    Returns:
        loss: Scalar loss value
        loss_dict: Dictionary with detailed losses
    """
    # Split RGB and Alpha
    pred_rgb = pred_img[:, :3]  # (B, 3, H, W)
    pred_alpha = pred_img[:, 3:4]  # (B, 1, H, W)
    target_rgb = target_img[:, :3]
    target_alpha = target_img[:, 3:4]

    # Compute RGB loss
    if loss_type == "l1":
        rgb_loss = F.l1_loss(pred_rgb, target_rgb, reduction="mean")
        alpha_loss = F.l1_loss(pred_alpha, target_alpha, reduction="mean")
    elif loss_type == "l2":
        rgb_loss = F.mse_loss(pred_rgb, target_rgb, reduction="mean")
        alpha_loss = F.mse_loss(pred_alpha, target_alpha, reduction="mean")
    elif loss_type == "huber":
        rgb_loss = F.huber_loss(pred_rgb, target_rgb, reduction="mean", delta=0.1)
        alpha_loss = F.huber_loss(pred_alpha, target_alpha, reduction="mean", delta=0.1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Weighted combination
    total_loss = rgb_weight * rgb_loss + alpha_weight * alpha_loss

    loss_dict = {
        "loss": total_loss.item(),
        "rgb_loss": rgb_loss.item(),
        "alpha_loss": alpha_loss.item(),
    }

    return total_loss, loss_dict


def generate_with_model(
    pipe,
    trans_vae,
    initial_latent,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    height,
    width,
    num_inference_steps,
    guidance_scale,
):
    """
    Generate image from initial latent using the model

    Returns:
        pred_img: (B, 4, H, W) predicted RGBA image in [0, 1]
    """
    batch_size = initial_latent.shape[0]
    device = initial_latent.device

    # Prepare latents
    latents = initial_latent.clone()

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Prepare latent image ids (for Flux positional encoding)
    latent_image_ids = _prepare_latent_image_ids(
        batch_size,
        height // pipe.vae_scale_factor // 2,
        width // pipe.vae_scale_factor // 2,
        device,
        pipe.transformer.dtype
    )

    # Pack latents
    latents = _pack_latents(
        latents,
        batch_size,
        latents.shape[1],
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor
    )

    # Denoising loop
    for i, t in enumerate(timesteps):
        # Expand latents for classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

        # Timestep embedding
        timestep = t.expand(latent_model_input.shape[0]).to(latent_model_input.dtype)

        # Prepare conditioning
        if guidance_scale > 1.0:
            prompt_embeds_input = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds])
            pooled_prompt_embeds_input = torch.cat([torch.zeros_like(pooled_prompt_embeds), pooled_prompt_embeds])
            text_ids_input = torch.cat([text_ids, text_ids])
            latent_image_ids_input = torch.cat([latent_image_ids, latent_image_ids])
        else:
            prompt_embeds_input = prompt_embeds
            pooled_prompt_embeds_input = pooled_prompt_embeds
            text_ids_input = text_ids
            latent_image_ids_input = latent_image_ids

        # Predict noise
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=torch.tensor([guidance_scale], device=device, dtype=latent_model_input.dtype),
            pooled_projections=pooled_prompt_embeds_input,
            encoder_hidden_states=prompt_embeds_input,
            txt_ids=text_ids_input,
            img_ids=latent_image_ids_input,
            return_dict=False,
        )[0]

        # Classifier free guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # Compute previous noisy sample
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Unpack latents
    latents = _unpack_latents(
        latents,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
        pipe.vae.config.latent_channels
    )

    # Decode to image space
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    with torch.no_grad():
        _, pred_rgba = trans_vae.decode(latents, aug=False)

    # pred_rgba is already in [0, 1]
    pred_rgba = pred_rgba.clamp(0, 1)

    return pred_rgba


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Prepare latent image position IDs for Flux"""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_ids = latent_image_ids.reshape(-1, 3)
    latent_image_ids = latent_image_ids.unsqueeze(0).repeat(batch_size, 1, 1)
    return latent_image_ids.to(device=device, dtype=dtype)


def _pack_latents(latents, batch_size, channels, height, width):
    """Pack latents for Flux (2x2 patch packing)"""
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)
    return latents


def _unpack_latents(latents, height, width, channels):
    """Unpack latents from Flux format"""
    batch_size = latents.shape[0]
    latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels, height, width)
    return latents


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
    logger.info(f"Training with PIXEL-SPACE LOSS")
    logger.info(f"  - RGB loss weight: {args.rgb_loss_weight}")
    logger.info(f"  - Alpha loss weight: {args.alpha_loss_weight}")
    logger.info(f"  - Loss type: {args.loss_type}")
    logger.info(f"  - Inference steps per training step: {args.num_inference_steps}")

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
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_config)
    pipe.transformer.print_trainable_parameters()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

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

                # Target image (ground truth)
                target_img = img_rgba.clone()  # (B, 4, H, W) in [0, 1]

                # Encode to latents
                with torch.no_grad():
                    if args.enable_cpu_offload:
                        pipe.vae.to(accelerator.device)
                        trans_vae.to(accelerator.device)

                    initial_latent = trans_vae.encode(
                        img_rgba=img_rgba,
                        img_rgb=img_rgb,
                        padded_img_rgb=padded_rgb,
                        use_offset=use_transparency_encoder,
                    )

                    # Encode prompts
                    if args.enable_cpu_offload:
                        pipe.text_encoder.to(accelerator.device)
                        pipe.text_encoder_2.to(accelerator.device)

                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        text_ids,
                    ) = pipe.encode_prompt(
                        prompt=captions,
                        prompt_2=None,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                    )

                    if args.enable_cpu_offload:
                        pipe.text_encoder.to("cpu")
                        pipe.text_encoder_2.to("cpu")
                        torch.cuda.empty_cache()

                # Generate image (this uses the trainable transformer)
                pred_img = generate_with_model(
                    pipe=pipe,
                    trans_vae=trans_vae,
                    initial_latent=initial_latent,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    height=height,
                    width=width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )

                # Compute loss in image space
                loss, loss_dict = compute_image_loss(
                    pred_img=pred_img,
                    target_img=target_img,
                    loss_type=args.loss_type,
                    rgb_weight=args.rgb_loss_weight,
                    alpha_weight=args.alpha_loss_weight,
                )

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(pipe.transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Free memory
                del img_rgba, img_rgb, padded_rgb, pred_img, target_img
                if args.enable_cpu_offload:
                    pipe.vae.to("cpu")
                    trans_vae.to("cpu")
                    torch.cuda.empty_cache()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss_dict["loss"],
                        "rgb_loss": loss_dict["rgb_loss"],
                        "alpha_loss": loss_dict["alpha_loss"],
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step,
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
            "training_mode": "pixel_space_loss",
            "rgb_loss_weight": args.rgb_loss_weight,
            "alpha_loss_weight": args.alpha_loss_weight,
            "loss_type": args.loss_type,
            "num_inference_steps": args.num_inference_steps,
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
