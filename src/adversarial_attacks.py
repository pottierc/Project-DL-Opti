"""
Adversarial Attack Utilities for CIFAR-10

This module contains utility functions for generating adversarial examples
and evaluating model robustness on the CIFAR-10 dataset.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def fgsm_attack(model: nn.Module, 
                images: torch.Tensor, 
                labels: torch.Tensor, 
                epsilon: float = 0.03) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    
    Args:
        model: PyTorch model to attack
        images: Input images tensor
        labels: True labels tensor
        epsilon: Attack strength (perturbation magnitude)
    
    Returns:
        Adversarial examples tensor
    """
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples
    sign_data_grad = images.grad.data.sign()
    perturbed_images = images + epsilon * sign_data_grad
    
    # Clip to maintain valid pixel values [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images


def pgd_attack(model: nn.Module,
               images: torch.Tensor,
               labels: torch.Tensor,
               epsilon: float = 0.03,
               alpha: float = 0.007,
               num_iter: int = 10) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) adversarial attack.
    
    Args:
        model: PyTorch model to attack
        images: Input images tensor
        labels: True labels tensor
        epsilon: Attack strength (L-infinity bound)
        alpha: Step size
        num_iter: Number of iterations
    
    Returns:
        Adversarial examples tensor
    """
    # Start with a copy of the original images
    adv_images = images.clone().detach()
    
    for _ in range(num_iter):
        adv_images.requires_grad = True
        
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Update adversarial images
        adv_images = adv_images + alpha * adv_images.grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + eta, min=0, max=1).detach()
    
    return adv_images


def evaluate_robustness(model: nn.Module,
                       data_loader: torch.utils.data.DataLoader,
                       attack_fn: callable,
                       device: torch.device,
                       **attack_kwargs) -> Tuple[float, float]:
    """
    Evaluate model robustness against adversarial attacks.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for test set
        attack_fn: Adversarial attack function
        device: Device to run computations on
        **attack_kwargs: Additional arguments for attack function
    
    Returns:
        Tuple of (clean_accuracy, adversarial_accuracy)
    """
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Clean accuracy
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            clean_correct += (predicted == labels).sum().item()
            
            # Generate adversarial examples
            adv_images = attack_fn(model, images, labels, **attack_kwargs)
            
            # Adversarial accuracy
            adv_outputs = model(adv_images)
            _, adv_predicted = torch.max(adv_outputs, 1)
            adv_correct += (adv_predicted == labels).sum().item()
            
            total += labels.size(0)
    
    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    
    return clean_acc, adv_acc


def compute_gradient_norm(model: nn.Module,
                         images: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient norm with respect to input images.
    
    Args:
        model: PyTorch model
        images: Input images tensor
        labels: True labels tensor
    
    Returns:
        Gradient norms tensor
    """
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    grad_norms = torch.norm(images.grad.data.view(images.size(0), -1), dim=1)
    
    return grad_norms