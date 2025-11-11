"""
Dataset loader for transparent image style fine-tuning
Supports loading transparent PNG images with alpha channels
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import json
from pathlib import Path
from lib_layerdiffuse.vae import pad_rgb
import torchvision.transforms as transforms


class TransparentImageDataset(Dataset):
    """
    Dataset for transparent images with text captions

    Expected directory structure:
    data_dir/
        images/
            image001.png
            image002.png
            ...
        metadata.jsonl  # Each line: {"file_name": "image001.png", "text": "caption text"}

    Or with individual caption files:
    data_dir/
        images/
            image001.png
            image001.txt
            image002.png
            image002.txt
            ...
    """

    def __init__(
        self,
        data_dir,
        resolution=1024,
        center_crop=True,
        random_flip=False,
        metadata_file="metadata.jsonl",
        use_padded_rgb=True,
    ):
        """
        Args:
            data_dir: Root directory containing images and captions
            resolution: Target image resolution (default: 1024)
            center_crop: Whether to center crop images
            random_flip: Whether to randomly flip images horizontally
            metadata_file: Name of the metadata file (JSONL format)
            use_padded_rgb: Whether to generate padded RGB for transparent regions
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.use_padded_rgb = use_padded_rgb

        # Load image-caption pairs
        self.samples = self._load_samples(metadata_file)

        print(f"Loaded {len(self.samples)} training samples from {data_dir}")

    def _load_samples(self, metadata_file):
        """Load image and caption pairs"""
        samples = []

        # Try loading from metadata.jsonl
        metadata_path = self.data_dir / metadata_file
        if metadata_path.exists():
            print(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    image_path = self.data_dir / "images" / data["file_name"]
                    if image_path.exists():
                        samples.append({
                            "image_path": str(image_path),
                            "caption": data["text"]
                        })
        else:
            # Fall back to loading from individual .txt files
            print(f"Metadata file not found, looking for individual caption files...")
            images_dir = self.data_dir / "images"
            if not images_dir.exists():
                images_dir = self.data_dir  # Images might be in root directory

            for img_file in images_dir.glob("*.png"):
                txt_file = img_file.with_suffix('.txt')
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    samples.append({
                        "image_path": str(img_file),
                        "caption": caption
                    })
                else:
                    # If no caption file, use filename as caption
                    caption = img_file.stem.replace('_', ' ')
                    samples.append({
                        "image_path": str(img_file),
                        "caption": caption
                    })

        if len(samples) == 0:
            raise ValueError(f"No valid samples found in {self.data_dir}")

        return samples

    def _load_image(self, image_path):
        """Load a transparent PNG image"""
        image = Image.open(image_path).convert('RGBA')
        return image

    def _prepare_image(self, image):
        """
        Prepare image for training
        Returns:
            img_rgba: RGBA tensor (C, H, W) with values in [0, 1]
            img_rgb: RGB tensor (C, H, W) with values in [-1, 1]
            padded_rgb: Padded RGB tensor (C, H, W) for transparent regions
        """
        # Resize and crop
        image = self._resize_and_crop(image)

        # Random horizontal flip
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy array
        img_np = np.array(image).astype(np.float32)  # (H, W, 4)

        # Generate padded RGB if needed
        if self.use_padded_rgb:
            padded_rgb_np = pad_rgb(img_np.astype(np.uint8))  # Returns float32
            padded_rgb = torch.from_numpy(padded_rgb_np).permute(2, 0, 1) / 255.0  # (C, H, W), [0, 1]
        else:
            padded_rgb = None

        # Convert to tensor
        img_rgba = torch.from_numpy(img_np).permute(2, 0, 1) / 255.0  # (4, H, W), [0, 1]

        # Extract RGB and apply normalization for VAE
        rgb = img_rgba[:3]  # (3, H, W)
        alpha = img_rgba[3:4]  # (1, H, W)

        # Premultiply alpha for RGB
        img_rgb = rgb * alpha  # (3, H, W), [0, 1]
        img_rgb = img_rgb * 2.0 - 1.0  # Normalize to [-1, 1] for VAE

        return img_rgba, img_rgb, padded_rgb

    def _resize_and_crop(self, image):
        """Resize and optionally center crop the image"""
        width, height = image.size

        # Resize
        if width < height:
            new_width = self.resolution
            new_height = int(height * self.resolution / width)
        else:
            new_height = self.resolution
            new_width = int(width * self.resolution / height)

        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Center crop if needed
        if self.center_crop:
            width, height = image.size
            left = (width - self.resolution) // 2
            top = (height - self.resolution) // 2
            right = left + self.resolution
            bottom = top + self.resolution
            image = image.crop((left, top, right, bottom))

        return image

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = self._load_image(sample["image_path"])

        # Prepare image tensors
        img_rgba, img_rgb, padded_rgb = self._prepare_image(image)

        # Get caption
        caption = sample["caption"]

        return {
            "img_rgba": img_rgba,  # (4, H, W), [0, 1]
            "img_rgb": img_rgb,    # (3, H, W), [-1, 1]
            "padded_rgb": padded_rgb if padded_rgb is not None else torch.zeros_like(img_rgb),  # (3, H, W), [0, 1]
            "caption": caption,
            "image_path": sample["image_path"]
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    img_rgba = torch.stack([item["img_rgba"] for item in batch])
    img_rgb = torch.stack([item["img_rgb"] for item in batch])
    padded_rgb = torch.stack([item["padded_rgb"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "img_rgba": img_rgba,
        "img_rgb": img_rgb,
        "padded_rgb": padded_rgb,
        "captions": captions,
        "image_paths": image_paths
    }


def create_dataloader(
    data_dir,
    batch_size=1,
    num_workers=4,
    resolution=1024,
    center_crop=True,
    random_flip=True,
    shuffle=True,
):
    """
    Create a DataLoader for transparent image training

    Args:
        data_dir: Root directory containing training data
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        resolution: Target image resolution
        center_crop: Whether to center crop images
        random_flip: Whether to randomly flip images
        shuffle: Whether to shuffle the dataset

    Returns:
        DataLoader instance
    """
    dataset = TransparentImageDataset(
        data_dir=data_dir,
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # Test the dataset loader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,  # Use 0 for debugging
        shuffle=False,
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Test loading one batch
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  img_rgba shape: {batch['img_rgba'].shape}")
        print(f"  img_rgb shape: {batch['img_rgb'].shape}")
        print(f"  padded_rgb shape: {batch['padded_rgb'].shape}")
        print(f"  captions: {batch['captions']}")
        print(f"  image_paths: {batch['image_paths']}")

        # Only test first batch
        break

    print("\nDataset test completed successfully!")
