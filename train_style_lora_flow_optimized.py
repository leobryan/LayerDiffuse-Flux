"""
Memory-optimized Flow Matching training for style LoRA
Includes latent caching, text embedding caching, 8-bit Adam, and aggressive memory management
"""

import argparse
import logging
import math
import os
import json
import gc
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


def log_memory(prefix=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"{prefix} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def free_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Memory-optimized Flow Matching training")

    # Model arguments
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--trans_vae", type=str, default="./models/TransparentVAE.pth")

    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--aspect_ratio_type", type=str, default="square",
                        choices=["square", "portrait", "landscape", "auto"])
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./style_lora_output_flow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None)

    # Flow Matching arguments
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # Generation arguments
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--disable_offset", action="store_true")

    # Memory optimization arguments
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="Cache VAE-encoded latents to save memory (recommended)"
    )
    parser.add_argument(
        "--cache_text_embeddings",
        action="store_true",
        help="Cache text embeddings when using same prompt (huge memory savings)"
    )
    parser.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Offload VAE and text encoders to CPU when not in use"
    )
    parser.add_argument(
        "--offload_text_encoder_to_cpu",
        action="store_true",
        default=True,
        help="Keep text encoders on CPU after encoding (default: True)"
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer to reduce memory"
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Enable xformers memory efficient attention"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0 to save memory)"
    )

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--log_memory_usage", action="store_true",
                        help="Log GPU memory usage")

    args = parser.parse_args()
    return args


def compute_flow_matching_loss(model_pred, target, weighting):
    """Compute Flow Matching loss with weighting"""
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


def cache_latents_and_embeddings(
    train_dataloader,
    pipe,
    trans_vae,
    accelerator,
    args,
    weight_dtype,
    use_transparency_encoder,
):
    """
    Pre-encode all images to latents and all prompts to embeddings
    This saves massive amounts of memory during training
    """
    logger.info("Caching latents and text embeddings...")

    latents_cache = []
    text_embeddings_cache = []
    unique_prompts = {}  # Cache unique prompts

    # Move models to GPU for encoding (always needed during caching, regardless of CPU offload setting)
    pipe.vae.to(accelerator.device)
    trans_vae.to(accelerator.device)
    pipe.text_encoder.to(accelerator.device)
    pipe.text_encoder_2.to(accelerator.device)

    progress_bar = tqdm(train_dataloader, desc="Caching", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in progress_bar:
            # Encode images to latents
            img_rgba = batch["img_rgba"].to(accelerator.device, dtype=torch.float32)
            img_rgb = batch["img_rgb"].to(accelerator.device, dtype=pipe.vae.dtype)
            padded_rgb = batch["padded_rgb"].to(accelerator.device, dtype=torch.float32)

            latents = trans_vae.encode(
                img_rgba=img_rgba,
                img_rgb=img_rgb,
                padded_img_rgb=padded_rgb,
                use_offset=use_transparency_encoder,
            ).to(dtype=weight_dtype)

            latents_cache.append(latents.cpu())  # Move to CPU to save GPU memory

            # Encode text (with caching for same prompts)
            captions = batch["captions"]
            batch_embeddings = []

            for caption in captions:
                if caption not in unique_prompts:
                    # First time seeing this prompt, encode it
                    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=[caption],
                        prompt_2=None,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                    )
                    # Store on CPU
                    unique_prompts[caption] = (
                        prompt_embeds.cpu(),
                        pooled_prompt_embeds.cpu(),
                        text_ids.cpu(),
                    )
                else:
                    # Reuse cached embeddings
                    pass

                batch_embeddings.append(caption)

            text_embeddings_cache.append(batch_embeddings)

            # Free memory
            del img_rgba, img_rgb, padded_rgb, latents
            if batch.get("prompt_embeds") is not None:
                del batch["prompt_embeds"]

    # Move models to CPU to free GPU memory
    if args.offload_text_encoder_to_cpu or args.enable_cpu_offload:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
    if args.enable_cpu_offload:
        pipe.vae.to("cpu")
        trans_vae.to("cpu")

    free_memory()

    logger.info(f"Cached {len(latents_cache)} batches")
    logger.info(f"Cached {len(unique_prompts)} unique text prompts")

    return latents_cache, text_embeddings_cache, unique_prompts


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

    # Log memory optimization settings
    logger.info("=" * 80)
    logger.info("MEMORY OPTIMIZATION SETTINGS")
    logger.info("=" * 80)
    logger.info(f"  Cache latents: {args.cache_latents}")
    logger.info(f"  Cache text embeddings: {args.cache_text_embeddings}")
    logger.info(f"  CPU offload: {args.enable_cpu_offload}")
    logger.info(f"  Offload text encoder: {args.offload_text_encoder_to_cpu}")
    logger.info(f"  8-bit Adam: {args.use_8bit_adam}")
    logger.info(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    logger.info(f"  XFormers: {args.enable_xformers}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info("=" * 80)

    if args.log_memory_usage:
        log_memory("Initial")

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

    if args.log_memory_usage:
        log_memory("After loading models")

    # Setup LoRA
    logger.info("Setting up LoRA adapter")

    if args.lora_target_modules is None:
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
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

    trainable_params = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.transformer.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Enable optimizations
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    if args.enable_xformers:
        try:
            pipe.transformer.enable_xformers_memory_efficient_attention()
            logger.info("XFormers enabled")
        except Exception as e:
            logger.warning(f"Could not enable XFormers: {e}")

    # Create noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
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
        num_workers=args.dataloader_num_workers,
    )

    # Set weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Cache latents and embeddings if requested
    latents_cache = None
    text_embeddings_cache = None
    unique_prompts = None

    if args.cache_latents or args.cache_text_embeddings:
        latents_cache, text_embeddings_cache, unique_prompts = cache_latents_and_embeddings(
            train_dataloader=train_dataloader,
            pipe=pipe,
            trans_vae=trans_vae,
            accelerator=accelerator,
            args=args,
            weight_dtype=weight_dtype,
            use_transparency_encoder=use_transparency_encoder,
        )

        if args.log_memory_usage:
            log_memory("After caching")

    # Optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            logger.info("Using 8-bit Adam optimizer")
        except ImportError:
            logger.warning("bitsandbytes not available, using standard AdamW")
            optimizer_class = torch.optim.AdamW
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
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

    # Move models to appropriate devices
    if not args.cache_latents and not args.enable_cpu_offload:
        pipe.vae.to(accelerator.device)
        trans_vae.to(accelerator.device)

    if not args.cache_text_embeddings and not args.offload_text_encoder_to_cpu:
        pipe.text_encoder.to(accelerator.device)
        pipe.text_encoder_2.to(accelerator.device)

    if args.log_memory_usage:
        log_memory("After preparation")

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
                batch_size = batch["img_rgba"].shape[0] if not args.cache_latents else latents_cache[step].shape[0]

                # Get latents (from cache or encode)
                if args.cache_latents:
                    model_input = latents_cache[step].to(accelerator.device, dtype=weight_dtype)
                    height, width = model_input.shape[2] * 16, model_input.shape[3] * 16  # Reverse VAE scaling
                else:
                    img_rgba = batch["img_rgba"].to(accelerator.device, dtype=torch.float32)
                    img_rgb = batch["img_rgb"].to(accelerator.device, dtype=pipe.vae.dtype)
                    padded_rgb = batch["padded_rgb"].to(accelerator.device, dtype=torch.float32)
                    height, width = img_rgba.shape[2], img_rgba.shape[3]

                    with torch.no_grad():
                        if args.enable_cpu_offload:
                            pipe.vae.to(accelerator.device)
                            trans_vae.to(accelerator.device)

                        model_input = trans_vae.encode(
                            img_rgba=img_rgba,
                            img_rgb=img_rgb,
                            padded_img_rgb=padded_rgb,
                            use_offset=use_transparency_encoder,
                        ).to(dtype=weight_dtype)

                        if args.enable_cpu_offload:
                            pipe.vae.to("cpu")
                            trans_vae.to("cpu")

                    del img_rgba, img_rgb, padded_rgb

                # Get text embeddings (from cache or encode)
                if args.cache_text_embeddings and unique_prompts is not None:
                    captions = text_embeddings_cache[step]
                    # Stack embeddings from cache
                    prompt_embeds_list = []
                    pooled_embeds_list = []
                    text_ids_list = []
                    for caption in captions:
                        pe, ppe, ti = unique_prompts[caption]
                        prompt_embeds_list.append(pe)
                        pooled_embeds_list.append(ppe)
                        text_ids_list.append(ti)

                    prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(accelerator.device)
                    pooled_prompt_embeds = torch.cat(pooled_embeds_list, dim=0).to(accelerator.device)
                    text_ids = text_ids_list[0].to(accelerator.device)  # All same
                else:
                    captions = batch["captions"]
                    with torch.no_grad():
                        if args.enable_cpu_offload or args.offload_text_encoder_to_cpu:
                            pipe.text_encoder.to(accelerator.device)
                            pipe.text_encoder_2.to(accelerator.device)

                        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                            prompt=captions,
                            prompt_2=None,
                            device=accelerator.device,
                            num_images_per_prompt=1,
                            max_sequence_length=512,
                        )

                        if args.offload_text_encoder_to_cpu:
                            pipe.text_encoder.to("cpu")
                            pipe.text_encoder_2.to("cpu")

                # Prepare latent image ids
                latent_image_ids = pipe._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # Sample noise and timesteps
                noise = torch.randn_like(model_input)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=batch_size,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=model_input.device)

                # Flow matching interpolation
                sigmas = get_sigmas(timesteps, noise_scheduler, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Pack latents
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

                # Compute loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas
                )
                target = noise - model_input
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
                optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory

                # Free memory
                del model_input, noise, model_pred, target, noisy_model_input
                del prompt_embeds, pooled_prompt_embeds, text_ids, latent_image_ids

                if step % 10 == 0:
                    free_memory()

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
                    if args.log_memory_usage:
                        logs["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
                        logs["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if args.log_memory_usage and global_step % 100 == 0:
                        log_memory(f"Step {global_step}")

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        logger.info(f"Saving checkpoint to {save_path}")
                        os.makedirs(save_path, exist_ok=True)

                        unwrapped_transformer = accelerator.unwrap_model(pipe.transformer)
                        pipe.transformer = unwrapped_transformer
                        pipe.save_lora_weights(save_path, safe_serialization=True)
                        pipe.transformer = accelerator.prepare(unwrapped_transformer)

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
        pipe.save_lora_weights(save_path, safe_serialization=True)

        metadata = {
            "base_model": args.base_model,
            "trans_vae": args.trans_vae,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "training_mode": "flow_matching_loss_memory_optimized",
            "weighting_scheme": args.weighting_scheme,
            "cache_latents": args.cache_latents,
            "cache_text_embeddings": args.cache_text_embeddings,
            "use_8bit_adam": args.use_8bit_adam,
            "training_steps": global_step,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"LoRA weights saved to {save_path}")

    if args.log_memory_usage:
        log_memory("Final")

    accelerator.end_training()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
