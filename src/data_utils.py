"""
CIFAR-10 Dataset Utilities

This module contains utility functions for loading and preprocessing
the CIFAR-10 dataset for adversarial robustness experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CIFAR-10 statistics
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


def get_cifar10_loaders(batch_size: int = 128,
                       num_workers: int = 2,
                       pin_memory: bool = True,
                       data_augmentation: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 data loaders for training and testing.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        data_augmentation: Whether to apply data augmentation to training set
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Define transforms
    if data_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, test_loader


def unnormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize CIFAR-10 images for visualization.
    
    Args:
        tensor: Normalized image tensor
    
    Returns:
        Unnormalized image tensor
    """
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean


def visualize_samples(images: torch.Tensor,
                     labels: torch.Tensor,
                     predictions: torch.Tensor = None,
                     num_samples: int = 8,
                     title: str = "CIFAR-10 Samples") -> None:
    """
    Visualize CIFAR-10 image samples.
    
    Args:
        images: Image tensor (normalized)
        labels: True labels tensor
        predictions: Predicted labels tensor (optional)
        num_samples: Number of samples to visualize
        title: Title for the plot
    """
    # Unnormalize images for visualization
    images = unnormalize_cifar10(images)
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    if predictions is not None:
        predictions = predictions.cpu().numpy()
    
    # Create subplot
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        # Transpose from CHW to HWC format
        img = np.transpose(images[i], (1, 2, 0))
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Create title with true and predicted labels
        true_class = CIFAR10_CLASSES[labels[i]]
        if predictions is not None:
            pred_class = CIFAR10_CLASSES[predictions[i]]
            color = 'green' if labels[i] == predictions[i] else 'red'
            axes[i].set_title(f'True: {true_class}\nPred: {pred_class}', 
                            color=color, fontsize=10)
        else:
            axes[i].set_title(f'True: {true_class}', fontsize=10)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def compare_clean_vs_adversarial(clean_images: torch.Tensor,
                                adv_images: torch.Tensor,
                                labels: torch.Tensor,
                                clean_preds: torch.Tensor,
                                adv_preds: torch.Tensor,
                                num_samples: int = 4) -> None:
    """
    Compare clean and adversarial images side by side.
    
    Args:
        clean_images: Clean image tensor
        adv_images: Adversarial image tensor
        labels: True labels tensor
        clean_preds: Clean predictions tensor
        adv_preds: Adversarial predictions tensor
        num_samples: Number of samples to visualize
    """
    # Unnormalize images
    clean_images = unnormalize_cifar10(clean_images)
    adv_images = unnormalize_cifar10(adv_images)
    
    clean_images = torch.clamp(clean_images, 0, 1).cpu().numpy()
    adv_images = torch.clamp(adv_images, 0, 1).cpu().numpy()
    
    labels = labels.cpu().numpy()
    clean_preds = clean_preds.cpu().numpy()
    adv_preds = adv_preds.cpu().numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        # Clean images (top row)
        clean_img = np.transpose(clean_images[i], (1, 2, 0))
        axes[0, i].imshow(clean_img)
        axes[0, i].axis('off')
        
        true_class = CIFAR10_CLASSES[labels[i]]
        clean_pred_class = CIFAR10_CLASSES[clean_preds[i]]
        color = 'green' if labels[i] == clean_preds[i] else 'red'
        axes[0, i].set_title(f'Clean\nTrue: {true_class}\nPred: {clean_pred_class}', 
                           color=color, fontsize=10)
        
        # Adversarial images (bottom row)
        adv_img = np.transpose(adv_images[i], (1, 2, 0))
        axes[1, i].imshow(adv_img)
        axes[1, i].axis('off')
        
        adv_pred_class = CIFAR10_CLASSES[adv_preds[i]]
        color = 'green' if labels[i] == adv_preds[i] else 'red'
        axes[1, i].set_title(f'Adversarial\nTrue: {true_class}\nPred: {adv_pred_class}', 
                           color=color, fontsize=10)
    
    plt.suptitle('Clean vs Adversarial Examples', fontsize=14)
    plt.tight_layout()
    plt.show()


def calculate_dataset_stats() -> Dict[str, Any]:
    """
    Calculate and return statistics about the CIFAR-10 dataset.
    
    Returns:
        Dictionary containing dataset statistics
    """
    # Load dataset without normalization
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    return {
        'num_classes': 10,
        'class_names': CIFAR10_CLASSES,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'image_shape': (3, 32, 32),
        'mean': CIFAR10_MEAN,
        'std': CIFAR10_STD
    }