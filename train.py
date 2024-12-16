import os
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomErasing
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import pandas as pd
from tqdm import tqdm
from timm import create_model
from sklearn.model_selection import train_test_split

# Define transforms

transform_train_unlabeled = transforms.Compose([
    transforms.RandomResizedCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_train = transforms.Compose([
    AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.RandomResizedCrop(384),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_val = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CutMix Function
def cutmix(images, labels, alpha=1.0):
    """
    Applies the CutMix augmentation.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        labels (torch.Tensor): Batch of labels (B,).
        alpha (float): Parameter for Beta distribution.
    
    Returns:
        Tuple of augmented images, new labels, and lambda value.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size, _, height, width = images.size()
    rand_index = torch.randperm(batch_size).to(images.device)
    labels_a = labels
    labels_b = labels[rand_index]

    # Random box coordinates
    rx = np.random.uniform(0, width)
    ry = np.random.uniform(0, height)
    rw = width * np.sqrt(1 - lam)
    rh = height * np.sqrt(1 - lam)

    x1 = int(max(rx - rw / 2, 0))
    x2 = int(min(rx + rw / 2, width))
    y1 = int(max(ry - rh / 2, 0))
    y2 = int(min(ry + rh / 2, height))

    # Create mixed image
    images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (height * width))
    
    return images, labels_a, labels_b, lam

# Mixup Function
def mixup(images, labels, alpha=1.0):
    """
    Applies the Mixup augmentation.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        labels (torch.Tensor): Batch of labels (B,).
        alpha (float): Parameter for Beta distribution.
    
    Returns:
        Tuple of mixed images, new labels_a, new labels_b, and lambda value.
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    rand_index = torch.randperm(batch_size).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[rand_index, :]
    labels_a = labels
    labels_b = labels[rand_index]
    return mixed_images, labels_a, labels_b, lam

# Random Erasing (Part of torchvision.transforms)
random_erasing = RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)

# Augmentation Pipeline
def apply_augmentations(images, labels, use_cutmix=False, use_mixup=False, use_random_erasing=False, alpha=1.0):
    """
    Applies the chosen augmentations (CutMix, Mixup, Random Erasing).
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W).
        labels (torch.Tensor): Batch of labels (B,).
        use_cutmix (bool): Whether to apply CutMix.
        use_mixup (bool): Whether to apply Mixup.
        use_random_erasing (bool): Whether to apply Random Erasing.
        alpha (float): Parameter for Beta distribution in CutMix and Mixup.
    
    Returns:
        Tuple of augmented images and corresponding labels.
    """
    if use_cutmix and use_mixup:
        raise ValueError("CutMix and Mixup cannot be applied simultaneously.")
    
    if use_cutmix:
        images, labels_a, labels_b, lam = cutmix(images, labels, alpha)
        return images, labels_a, labels_b, lam
    elif use_mixup:
        images, labels_a, labels_b, lam = mixup(images, labels, alpha)
        return images, labels_a, labels_b, lam
    elif use_random_erasing:
        images = torch.stack([random_erasing(image) for image in images])
        return images, labels
    else:
        return images, labels

class CustomLabeledDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx, 1])  # Ensure label is an integer
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom dataset for flat unlabeled data
class UnlabeledDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

csv_path = 'train_labeled.csv'
image_root = 'train/labeled'
unlabeled_folder_path = 'train/unlabeled'
# Checkpoint paths
last_checkpoint_path = 'last_checkpoint.pth'
best_checkpoint_path = 'best_checkpoint.pth'
pretrained = None

# Initialize variables for tracking best validation loss
best_val_loss = float('inf')
batch_size = 64
num_workers = 16
pseudo_label_threshold = 0.95

full_data = pd.read_csv(csv_path)
train_data, val_data = train_test_split(full_data, test_size=0.1, stratify=full_data['id'], random_state=42)

# Save split CSVs for tracking (optional)
train_csv = 'train_split.csv'
val_csv = 'val_split.csv'
train_data.to_csv(train_csv, index=False)
val_data.to_csv(val_csv, index=False)

# Create datasets
train_dataset = CustomLabeledDataset(csv_file=train_csv, root_dir=image_root, transform=transform_train)
val_dataset = CustomLabeledDataset(csv_file=val_csv, root_dir=image_root, transform=transform_val)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

unlabeled_dataset = UnlabeledDataset(unlabeled_folder_path, transform=transform_train_unlabeled)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create model
model = create_model('deit3_large_patch16_384', pretrained=True, num_classes=135)
model = model.to('cuda')

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded.")
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded.")
    last_epoch = checkpoint.get('epoch', -1)
    print(f"Checkpoint loaded: Last epoch = {last_epoch}")
    return last_epoch


if pretrained is not None:
    last_epoch = load_checkpoint(pretrained, model, optimizer, scheduler)
    scheduler.last_epoch = last_epoch

# Temporary folder for pseudo-labeled data
pseudo_label_folder = './pseudo_labels'
if os.path.exists(pseudo_label_folder):
    shutil.rmtree(pseudo_label_folder)
os.makedirs(pseudo_label_folder)

# Pseudo-labeling function
def generate_pseudo_labels_to_disk(model, unlabeled_loader, threshold=0.9):
    model.eval()
    pseudo_folder = os.path.join(pseudo_label_folder, 'pseudo_labeled')
    if os.path.exists(pseudo_label_folder):
        shutil.rmtree(pseudo_label_folder)
    os.makedirs(pseudo_folder, exist_ok=True)

    for class_idx in range(0, 135):
        os.makedirs(os.path.join(pseudo_folder, str(class_idx)), exist_ok=True)

    with torch.no_grad():
        for images, paths in tqdm(unlabeled_loader, desc="Generating Pseudo Labels"):
            images = images.to('cuda')
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)

            # Save high-confidence predictions
            for path, prob, label in zip(paths, max_probs, pseudo_labels):
                if prob > threshold:
                    image = Image.open(path).convert("RGB")
                    class_folder = os.path.join(pseudo_folder, str(label.item()))
                    os.makedirs(class_folder, exist_ok=True)
                    image.save(os.path.join(class_folder, os.path.basename(path)))

    return pseudo_folder
# Training supervised
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Train on labeled data
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Labeled Data"):
        images, labels = images.to('cuda'), labels.to('cuda')

        # Apply CutMix or Mixup
        use_cutmix, use_mixup = False, True  # Choose one
        if use_cutmix or use_mixup:
            images, labels_a, labels_b, lam = apply_augmentations(
                images, labels, use_cutmix=use_cutmix, use_mixup=use_mixup, use_random_erasing=True, alpha=1.0
            )

        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss for CutMix or Mixup
        if use_cutmix or use_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Supervised Loss: {running_loss / len(train_loader):.4f}")
    # Validate
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save the last checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_acc,
    }, last_checkpoint_path)
    print(f"Last checkpoint saved at: {last_checkpoint_path}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, best_checkpoint_path)
        print(f"Best checkpoint saved with validation loss: {best_val_loss:.4f} at: {best_checkpoint_path}")

    # Step scheduler
    scheduler.step()

# Training loop
epochs = 150
pseudo_label_loss_weight = 0.5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Train on labeled data
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Labeled Data"):
        images, labels = images.to('cuda'), labels.to('cuda')

        # Apply CutMix or Mixup
        use_cutmix, use_mixup = False, True  # Choose one
        if use_cutmix or use_mixup:
            images, labels_a, labels_b, lam = apply_augmentations(
                images, labels, use_cutmix=use_cutmix, use_mixup=use_mixup, use_random_erasing=True, alpha=1.0
            )
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss for CutMix or Mixup
        if use_cutmix or use_mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Supervised Loss: {running_loss / len(train_loader):.4f}")

    # Generate pseudo-labels
    pseudo_folder = generate_pseudo_labels_to_disk(model, unlabeled_loader, threshold=pseudo_label_threshold)

    # Train on pseudo-labeled data
    pseudo_dataset = ImageFolder(pseudo_folder, transform=transform_train, allow_empty=True)
    pseudo_loader = DataLoader(pseudo_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    running_pseudo_loss = 0.0

    for images, labels in tqdm(pseudo_loader, desc=f"Epoch {epoch+1}/{epochs} - Pseudo-Labeled Data"):
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        pseudo_loss = criterion(outputs, labels)
        (pseudo_label_loss_weight * pseudo_loss).backward()
        optimizer.step()
        running_pseudo_loss += pseudo_loss.item()

    print(f"Pseudo-Labeled Loss: {running_pseudo_loss / len(pseudo_loader):.4f}")

    # Validate
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    # Save the last checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_accuracy': val_acc,
    }, last_checkpoint_path)
    print(f"Last checkpoint saved at: {last_checkpoint_path}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
        }, best_checkpoint_path)
        print(f"Best checkpoint saved with validation loss: {best_val_loss:.4f} at: {best_checkpoint_path}")

    # Step scheduler
    scheduler.step()

