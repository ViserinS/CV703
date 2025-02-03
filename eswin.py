#!/usr/bin/env python3
"""
Train a Vision Transformer model (Swin or ESwin) on one of two datasets:
- ImageWoof2-160 only
- Concatenation of ImageWoof2-160, FGVC Aircraft, Flowers-102

Usage:
    python train_transformer.py --dataset [imagewoof | all] --model [swin | eswin | both]
"""

import os
import time
import argparse
import random
import numpy as np
import urllib.request
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from collections import Counter
import pandas as pd

# For FLOPs calculation
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    pass

# Hugging Face Swin
from transformers import SwinConfig, SwinForImageClassification, SwinModel
from transformers.models.swin.modeling_swin import SwinPatchEmbeddings

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ======================
# Directory config
# ======================
DATA_DIR = './data'

# ImageWoof
IMAGEWOOF_TGZ_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
IMAGEWOOF_TGZ_PATH = os.path.join(DATA_DIR, 'imagewoof2-160.tgz')
IMAGEWOOF_EXTRACT_PATH = os.path.join(DATA_DIR, 'imagewoof2-160')

# FGVC Aircraft & Flowers-102
FGVC_AIRCRAFT_ROOT = os.path.join(DATA_DIR)
FLOWERS102_ROOT = os.path.join(DATA_DIR, 'flowers-102')


# =====================================================
# 1. Data Preparation & Dataloader
# =====================================================
def download_imagewoof(url, download_path, extract_path):
    if not os.path.exists(os.path.dirname(download_path)):
        os.makedirs(os.path.dirname(download_path), exist_ok=True)

    if not os.path.exists(extract_path):
        print("Downloading ImageWoof2 dataset...")
        urllib.request.urlretrieve(url, download_path)
        print("Extracting ImageWoof2...")
        with tarfile.open(download_path, 'r:gz') as tar:
            tar.extractall(path=DATA_DIR)
        print("Done.")
    else:
        print("ImageWoof2 is already downloaded.")


def offset_labels(dataset, start_label):
    """
    Offset a dataset's labels by `start_label` to avoid overlap when concatenating multiple datasets.
    
    This function works with datasets that either have a `samples` attribute (e.g., ImageFolder)
    or not (e.g., torchvision's FGVCAircraft). In the latter case, a wrapper dataset is created
    that applies the label mapping during __getitem__.
    """
    # Gather all labels
    all_labels = []
    for _, lbl in dataset:
        all_labels.append(lbl)
    
    unique_labels = set(all_labels)
    num_unique = len(unique_labels)
    
    # Create a remapping dictionary: original label -> new label
    sorted_labels = sorted(unique_labels)
    remap = {old_label: start_label + i for i, old_label in enumerate(sorted_labels)}
    
    # If the dataset has 'samples', update it directly.
    if hasattr(dataset, 'samples'):
        for idx, (path, lbl) in enumerate(dataset.samples):
            dataset.samples[idx] = (path, remap[lbl])
        if hasattr(dataset, 'targets'):
            dataset.targets = [remap[lbl] for lbl in dataset.targets]
        new_dataset = dataset
    else:
        # For datasets without 'samples', wrap the dataset to apply the remap on the fly.
        class OffsetDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, mapping):
                self.base_dataset = base_dataset
                self.mapping = mapping

            def __getitem__(self, index):
                x, y = self.base_dataset[index]
                return x, self.mapping[y]

            def __len__(self):
                return len(self.base_dataset)

        new_dataset = OffsetDataset(dataset, remap)
    
    return new_dataset, start_label + num_unique

class ImageFolderOffset(ImageFolder):
    """
    Same as ImageFolder, but adds `start_label` to every returned label.
    """
    def __init__(self, root, transform=None, start_label=0):
        super().__init__(root=root, transform=transform)
        self.start_label = start_label

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # Offset the label
        return sample, target + self.start_label

def get_imagewoof(transform, batch_size=32):
    """
    Download (if needed) and return train/val dataloaders for ImageWoof2-160
    plus the number of classes in that dataset.
    """
    download_imagewoof(IMAGEWOOF_TGZ_URL, IMAGEWOOF_TGZ_PATH, IMAGEWOOF_EXTRACT_PATH)

    train_dir = os.path.join(IMAGEWOOF_EXTRACT_PATH, 'imagewoof2-160', 'train')
    valid_dir = os.path.join(IMAGEWOOF_EXTRACT_PATH, 'imagewoof2-160', 'val')

    train_ds = ImageFolder(root=train_dir, transform=transform)
    val_ds   = ImageFolder(root=valid_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(train_ds.classes)

def get_all_datasets(transform, batch_size=32):
    IMAGEWOOF_OFFSET = 0      # 10 classes, so next offset is 10
    FGVC_OFFSET = 10          # 100 classes, so next offset is 110
    FLOWERS_OFFSET = 110      # 102 classes, so next offset would be 212

    # 1) Load ImageWoof
    wtrain_dir = os.path.join(IMAGEWOOF_EXTRACT_PATH, 'imagewoof2-160', 'train')
    wvalid_dir = os.path.join(IMAGEWOOF_EXTRACT_PATH, 'imagewoof2-160', 'val')
    imagewoof_train = ImageFolderOffset(
        root=wtrain_dir, 
        transform=transform,
        start_label=IMAGEWOOF_OFFSET  # 0
    )
    imagewoof_val = ImageFolderOffset(
        root=wvalid_dir, 
        transform=transform,
        start_label=IMAGEWOOF_OFFSET  # 0
    )
    
    # 2) FGVC with start_label = 10
    from torchvision.datasets import FGVCAircraft
    class ModifiedFGVCAircraft(FGVCAircraft):
        def __init__(self, root, split='trainval', transform=None, download=False, startlabel=0):
            super().__init__(root=root, split=split, download=download, transform=transform)
            self.startlabel = startlabel
        
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            return x, y + self.startlabel

    fgvc_trainval = ModifiedFGVCAircraft(
        root=FGVC_AIRCRAFT_ROOT,
        split='trainval',
        transform=transform,
        download=True,
        startlabel=FGVC_OFFSET  # 10
    )
    fgvc_test = ModifiedFGVCAircraft(
        root=FGVC_AIRCRAFT_ROOT,
        split='test',
        transform=transform,
        download=True,
        startlabel=FGVC_OFFSET  # 10
    )
    
    # 3) Flowers-102 with start_label=110
    from torchvision.datasets import Flowers102
    class ModifiedFlowers102(Flowers102):
        def __init__(self, root, split='train', transform=None, download=False, startlabel=0):
            super().__init__(root=root, split=split, download=download, transform=transform)
            self.startlabel = startlabel
        
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            return x, y + self.startlabel

    flowers_train = ModifiedFlowers102(
        root=FLOWERS102_ROOT,
        split='train',
        transform=transform,
        download=True,
        startlabel=FLOWERS_OFFSET  # 110
    )
    flowers_test = ModifiedFlowers102(
        root=FLOWERS102_ROOT,
        split='test',
        transform=transform,
        download=True,
        startlabel=FLOWERS_OFFSET  # 110
    )

    # Concat them: train splits go together, test/val splits go together
    train_ds = ConcatDataset([imagewoof_train, fgvc_trainval, flowers_train])
    val_ds   = ConcatDataset([imagewoof_val, fgvc_test, flowers_test])

    # The total number of classes
    # = 10 (ImageWoof) + 100 (FGVC) + 102 (Flowers) = 212
    # but to be safe we can also compute it from train_ds:
    all_labels = []
    for _, lbl in train_ds:
        all_labels.append(lbl)
    num_classes = max(all_labels) + 1  # should be 212

    # Wrap in DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes


# =====================================================
# 2. Swin & ESwin Definitions
# =====================================================

class TwoLayerMLP(nn.Module):
    """
    A two-layer MLP (with LayerNorm & Dropout) that mirrors the
    gold-standard code's classifier block.
    """
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SwinCustomClassifier(nn.Module):
    """
    Wraps a HuggingFace SwinModel but replaces the top classification layer
    with our custom two-layer MLP head.
    """
    def __init__(self, base_model, mlp_head):
        super().__init__()
        self.swin = base_model
        self.classifier = mlp_head

    def forward(self, x):
        outputs = self.swin(x)
        # outputs.pooler_output is shape [batch_size, hidden_size]
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

def create_swin_model(num_classes=10, variant="microsoft/swin-base-patch4-window7-224"):
    """
    Loads a Swin-base model, strips its built-in classifier,
    and attaches the 2-layer MLP head from the gold-standard code.
    """
    # 1) Load base Swin with desired config
    config = SwinConfig.from_pretrained(variant)
    config.num_labels = num_classes  
    base_model = SwinModel.from_pretrained(variant, config=config, ignore_mismatched_sizes=True)

    # 2) Build the 2-layer MLP head
    mlp_head = TwoLayerMLP(hidden_size=config.hidden_size, num_classes=num_classes)

    # 3) Wrap them together
    model = SwinCustomClassifier(base_model, mlp_head)
    return model

class ConvolutionalStem(nn.Module):
    """
    A simple multi-layer convolutional stem.
    Although ESwin replaces the simple patch embedding with a convolutional block,
    it still “patchifies” the image by flattening spatial dimensions afterward.
    """
    def __init__(self, in_chans=3, embed_dim=128):
        super().__init__()
        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_chans),
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.c3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        return x

class ConvolutionalPatchEmbeddings(nn.Module):
    """
    Custom patch embedding module for ESwin.
    It replaces the standard patch embedding with a convolutional stem.
    """
    def __init__(self, orig_module, norm_layer=None):
        super().__init__()
        self.num_channels = orig_module.num_channels  # e.g. typically 3
        self.image_size = orig_module.image_size      # e.g. 224

        # Instead of directly accessing orig_module.embed_dim, use the projection's out_channels.
        self.embed_dim = getattr(orig_module, "embed_dim", orig_module.projection.out_channels)
        
        # Some versions may also have a patch_size attribute; if so, copy it.
        self.patch_size = getattr(orig_module, "patch_size", None)
        self.conv_stem = ConvolutionalStem(in_chans=self.num_channels, embed_dim=self.embed_dim)
        self.norm = norm_layer  # This could be a LayerNorm; if None, no normalization is applied.

    def forward(self, pixel_values: torch.Tensor):
        # Apply the convolutional stem
        x = self.conv_stem(pixel_values)  # shape: (B, embed_dim, H, W)
        b, c, h, w = x.shape

        # Flatten spatial dimensions: (B, H*W, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)

        # Return both the embeddings and the output dimensions (as expected by the Swin model)
        return x, (h, w)

class Downsample(nn.Module):
    """
    Downsampling module for ESwin.
    
    This module expects an input token sequence of shape (B, H*W, C) where H and W are the
    spatial dimensions as defined by self.input_resolution. It reshapes the tokens into a 2D 
    feature map, applies a convolution (that doubles the channel dimension), a max pooling 
    (that reduces H and W by about 2), and a normalization layer.
    
    The forward() method is defined to accept extra arguments (via *args, **kwargs) so that if 
    the calling code passes additional parameters (e.g. an updated resolution), they are simply 
    ignored.
    
    Args:
        input_resolution (tuple[int]): The spatial resolution (H, W) of the input feature map.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer to use (default: nn.LayerNorm).
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # (H, W)
        self.dim = dim
        # Convolution increases channels from dim to 2*dim.
        self.conv = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1)
        # Max pooling reduces spatial resolution by roughly a factor of 2.
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Normalization on the new channels.
        self.norm = norm_layer(2 * dim)

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x (Tensor): Input tensor of shape (B, H*W, C) where H*W matches the product of 
                        self.input_resolution.
            *args, **kwargs: Extra parameters that may be passed by the calling code.
            
        Returns:
            Tensor: Downsampled tensor of shape (B, (H//2)*(W//2), 2*C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"Input feature has wrong size: expected {H*W} tokens, got {L}.")
        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"Input resolution ({H}, {W}) should be even for downsampling.")
        
        # Reshape from (B, H*W, C) to (B, H, W, C).
        x = x.view(B, H, W, C)
        # Permute to (B, C, H, W) for convolution.
        x = x.permute(0, 3, 1, 2)
        # Apply convolution to increase channels.
        x = self.conv(x)
        # Permute back to (B, H, W, 2*C) for normalization.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Permute to (B, 2*C, H, W) for pooling.
        x = x.permute(0, 3, 1, 2)
        # Apply max pooling to reduce spatial resolution.
        x = self.pool(x)
        # Reshape the feature map back into a token sequence: (B, new_H*new_W, 2*C).
        B, new_C, new_H, new_W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, new_H * new_W, new_C)
        return x

def create_eswin_model(num_classes=10, variant="microsoft/swin-base-patch4-window7-224"):
    from transformers import SwinConfig, SwinModel
    
    config = SwinConfig.from_pretrained(variant)
    config.num_labels = num_classes
    base_model = SwinModel.from_pretrained(variant, config=config, ignore_mismatched_sizes=True)

    # Replace the patch embeddings with your conv stem
    orig_patch_embed = base_model.embeddings.patch_embeddings
    old_norm = getattr(orig_patch_embed, "norm", None)
    new_patch_embed = ConvolutionalPatchEmbeddings(orig_patch_embed, norm_layer=old_norm)
    base_model.embeddings.patch_embeddings = new_patch_embed

    # Figure out the post-stem resolution via a dummy forward pass
    img_size = getattr(orig_patch_embed, "image_size", (224, 224))
    if isinstance(img_size, int):
        img_size = (img_size, img_size)

    dummy = torch.zeros(1, 3, img_size[0], img_size[1])  # On CPU is fine
    with torch.no_grad():
        stem_out = new_patch_embed.conv_stem(dummy)
        # shape: (B, embed_dim, newH, newW)
        _, _, newH, newW = stem_out.shape

    input_res = (newH, newW)
    stage_dim = new_patch_embed.embed_dim

    # Update each stage's input_resolution + downsample
    for i, layer in enumerate(base_model.encoder.layers):
        if hasattr(layer, 'input_resolution'):
            layer.input_resolution = input_res

        # If not the last stage, add your custom Downsample
        if i < len(base_model.encoder.layers) - 1:
            layer.downsample = Downsample(input_resolution=input_res, dim=stage_dim)

        # Next stage is half the spatial size
        input_res = (input_res[0] // 2, input_res[1] // 2)
        stage_dim *= 2

    # Attach your 2-layer MLP head
    mlp_head = TwoLayerMLP(hidden_size=config.hidden_size, num_classes=num_classes)
    model = SwinCustomClassifier(base_model, mlp_head)
    return model



# =====================================================
# 3. Training Loop (unchanged except naming)
# =====================================================
def train_and_validate(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    num_epochs=20,
    patience=3,
    log_file="training_log.csv",
    best_model_path="best_model.pth",
    scheduler=None
):
    model = model.to(device)
    best_loss = float('inf')
    stopping_counter = 0

    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []

    overall_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # --------------------
        # Training
        # --------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}]", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_accuracy = 100.0 * val_correct / val_total

        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        epoch_list.append(epoch + 1)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)

        if scheduler is not None:
            scheduler.step()

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            stopping_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Early stopping triggered!")
                break

    total_training_time = time.time() - overall_start_time
    print(f"Total training time: {total_training_time:.2f}s")

    df = pd.DataFrame({
        "Epoch": epoch_list,
        "Train Loss": train_loss_list,
        "Train Accuracy": train_acc_list,
        "Val Loss": val_loss_list,
        "Val Accuracy": val_acc_list
    })
    df.to_csv(log_file, index=False)
    return df


# =====================================================
# 4. Model Info: Number of Params & FLOPs
# =====================================================
class FLOPsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        # If the model returns a tuple (e.g., (logits, aux_info)), return just the logits
        if isinstance(output, tuple):
            return output[0]
        return output


def get_model_info(model, input_res=(3, 224, 224)):
    """
    Returns (num_params, flops_str)
    - num_params: integer number of parameters
    - flops_str: string representing GFLOPs
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate FLOPs (if ptflops is available)
    if PTFLOPS_AVAILABLE:
        with torch.no_grad():
            # Wrap the model so that it returns a single tensor
            wrapped_model = FLOPsWrapper(model)
            macs, _ = get_model_complexity_info(wrapped_model, input_res, verbose=False)
        # macs is in GMAC; report as GFLOPs (approx.)
        flops_str = f"{macs} GFLOPs (approx.)"
    else:
        flops_str = "FLOPs: ptflops not installed"

    return num_params, flops_str

# =====================================================
# 5. Main
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Swin or ESwin Transformer on either ImageWoof or a combined dataset.")
    parser.add_argument("--dataset", type=str, default="imagewoof",
                        choices=["imagewoof", "all"],
                        help="Choose which dataset(s) to train on.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for optimizer.")
    parser.add_argument("--model", type=str, default="swin",
                        choices=["swin", "eswin", "both"],
                        help="Which model to train: 'swin', 'eswin', or 'both'.")
    parser.add_argument("--save_dir", type=str, default="./output")
    return parser.parse_args()


def main():
    args = parse_args()
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == "all":
        print("Loading ALL datasets: ImageWoof + FGVC Aircraft + Flowers-102")
        train_loader, val_loader, num_classes = get_all_datasets(
            transform=transform,
            batch_size=args.batch_size
        )
    else:
        print("Loading ImageWoof dataset only...")
        train_loader, val_loader, num_classes = get_imagewoof(
            transform=transform,
            batch_size=args.batch_size
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.model == "swin":
        print("\n==== Creating Swin Transformer ====")
        model_swin = create_swin_model(num_classes=num_classes)
        # Print model info
        swin_params, swin_flops = get_model_info(model_swin)
        print(f"Swin: # Params = {swin_params:,}, FLOPs = {swin_flops}")

        optimizer_swin = AdamW(model_swin.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler_swin = torch.optim.lr_scheduler.OneCycleLR(
                                    optimizer_swin,
                                    max_lr=1e-4, # or something close to your max in the first script
                                    total_steps=args.epochs * len(train_loader),
                                    pct_start=0.05,
                                    anneal_strategy='cos'
                                )
        train_and_validate(
            model=model_swin,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_swin,
            device=device,
            num_epochs=args.epochs,
            patience=5,
            log_file=os.path.join(SAVE_DIR, "swin_training_log.csv"),
            best_model_path=os.path.join(SAVE_DIR, "swin_best_model.pth"),
            scheduler=scheduler_swin
        )

    elif args.model == "eswin":
        print("\n==== Creating ESwin Transformer ====")
        model_eswin = create_eswin_model(num_classes=num_classes)
        # Print model info
        eswin_params, eswin_flops = get_model_info(model_eswin)
        print(f"ESwin: # Params = {eswin_params:,}, FLOPs = {eswin_flops}")

        optimizer_eswin = AdamW(model_eswin.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler_eswin = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer_eswin,
                            max_lr=1e-4, # or something close to your max in the first script
                            total_steps=args.epochs * len(train_loader),
                            pct_start=0.05,
                            anneal_strategy='cos'
                        )
        train_and_validate(
            model=model_eswin,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_eswin,
            device=device,
            num_epochs=args.epochs,
            patience=5,
            log_file=os.path.join(SAVE_DIR, "eswin_training_log.csv"),
            best_model_path=os.path.join(SAVE_DIR, "eswin_best_model.pth"),
            scheduler=scheduler_eswin
        )

    else:  # args.model == "both"
        print("\n==== Creating Swin Transformer ====")
        model_swin = create_swin_model(num_classes=num_classes)
        swin_params, swin_flops = get_model_info(model_swin)
        print(f"Swin: # Params = {swin_params:,}, FLOPs = {swin_flops}")

        print("\n==== Creating ESwin Transformer ====")
        model_eswin = create_eswin_model(num_classes=num_classes)
        eswin_params, eswin_flops = get_model_info(model_eswin)
        print(f"ESwin: # Params = {eswin_params:,}, FLOPs = {eswin_flops}")

        # Compare them
        param_diff = eswin_params - swin_params
        print(f"\n*** Model Parameter Difference ***")
        print(f"ESwin minus Swin = {param_diff:,} params")

        print(f"*** For FLOPs, approximate difference ***")
        print(f"(Swin)  {swin_flops}")
        print(f"(ESwin) {eswin_flops}")

        # Train both sequentially
        print("\n>>> Training Swin <<<")
        optimizer_swin = AdamW(model_swin.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler_swin = torch.optim.lr_scheduler.OneCycleLR(
                                    optimizer_swin,
                                    max_lr=1e-4, # or something close to your max in the first script
                                    total_steps=args.epochs * len(train_loader),
                                    pct_start=0.05,
                                    anneal_strategy='cos'
                                )
        train_and_validate(
            model=model_swin,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_swin,
            device=device,
            num_epochs=args.epochs,
            patience=5,
            log_file=os.path.join(SAVE_DIR, "swin_training_log.csv"),
            best_model_path=os.path.join(SAVE_DIR, "swin_best_model.pth"),
            scheduler=scheduler_swin
        )

        print("\n>>> Training ESwin <<<")
        optimizer_eswin = AdamW(model_eswin.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler_eswin = torch.optim.lr_scheduler.OneCycleLR(
                                    optimizer_eswin,
                                    max_lr=1e-4, # or something close to your max in the first script
                                    total_steps=args.epochs * len(train_loader),
                                    pct_start=0.05,
                                    anneal_strategy='cos'
                                )
        train_and_validate(
            model=model_eswin,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer_eswin,
            device=device,
            num_epochs=args.epochs,
            patience=5,
            log_file=os.path.join(SAVE_DIR, "eswin_training_log.csv"),
            best_model_path=os.path.join(SAVE_DIR, "eswin_best_model.pth"),
            scheduler=scheduler_eswin
        )


if __name__ == "__main__":
    main()
