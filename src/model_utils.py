"""
Model Utilities for Adversarial Robustness Experiments

This module contains utility functions for loading, training, and evaluating
deep learning models for adversarial robustness experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from typing import Tuple, Dict, Any, Optional
import os


def get_resnet18_cifar10(pretrained: bool = True, 
                        num_classes: int = 10) -> nn.Module:
    """
    Get a ResNet-18 model adapted for CIFAR-10.
    
    Args:
        pretrained: Whether to use pretrained ImageNet weights
        num_classes: Number of output classes (10 for CIFAR-10)
    
    Returns:
        ResNet-18 model adapted for CIFAR-10
    """
    if pretrained:
        model = models.resnet18(weights="DEFAULT")
    else:
        model = models.resnet18(weights=None)
    
    # Modify the final fully connected layer for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def train_model(model: nn.Module,
               train_loader: torch.utils.data.DataLoader,
               test_loader: torch.utils.data.DataLoader,
               device: torch.device,
               epochs: int = 10,
               lr: float = 0.01,
               weight_decay: float = 1e-4,
               save_path: Optional[str] = None) -> Dict[str, list]:
    """
    Train a model on the training data.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Test data loader for evaluation
        device: Device to run training on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        save_path: Path to save the trained model
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, 
                         momentum=0.9, weight_decay=weight_decay)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation phase
        test_acc = evaluate_model(model, test_loader, device)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}')
    
    # Save model if path provided
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'epochs': epochs,
            'lr': lr
        }, save_path)
        print(f'Model saved to {save_path}')
    
    return history


def evaluate_model(model: nn.Module,
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> float:
    """
    Evaluate model accuracy on given data.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
    
    Returns:
        Accuracy as a float
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def load_model(model: nn.Module,
               model_path: str,
               device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a saved model.
    
    Args:
        model: Model architecture (should match saved model)
        model_path: Path to saved model
        device: Device to load model on
    
    Returns:
        Tuple of (loaded_model, metadata)
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    metadata = {
        'history': checkpoint.get('history', {}),
        'epochs': checkpoint.get('epochs', 'unknown'),
        'lr': checkpoint.get('lr', 'unknown')
    }
    
    return model, metadata


def adversarial_training(model: nn.Module,
                        train_loader: torch.utils.data.DataLoader,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        attack_fn: callable,
                        epochs: int = 10,
                        lr: float = 0.01,
                        attack_prob: float = 0.5,
                        **attack_kwargs) -> Dict[str, list]:
    """
    Train a model with adversarial training.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device to run training on
        attack_fn: Adversarial attack function
        epochs: Number of training epochs
        lr: Learning rate
        attack_prob: Probability of using adversarial examples in training
        **attack_kwargs: Additional arguments for attack function
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'adv_test_acc': []
    }
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Randomly decide whether to use adversarial examples
            if torch.rand(1).item() < attack_prob:
                # Generate adversarial examples
                inputs = attack_fn(model, inputs, labels, **attack_kwargs)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        test_acc = evaluate_model(model, test_loader, device)
        
        # Adversarial evaluation (simplified)
        model.eval()
        adv_correct = 0
        adv_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                adv_inputs = attack_fn(model, inputs, labels, **attack_kwargs)
                
                outputs = model(adv_inputs)
                _, predicted = torch.max(outputs, 1)
                adv_total += labels.size(0)
                adv_correct += (predicted == labels).sum().item()
                
                # Only evaluate on a subset for speed
                if adv_total >= 1000:
                    break
        
        adv_test_acc = adv_correct / adv_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['adv_test_acc'].append(adv_test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f}, '
              f'Adv Test Acc: {adv_test_acc:.4f}')
    
    return history


def get_model_summary(model: nn.Module, input_size: tuple = (3, 32, 32)) -> None:
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (channels, height, width)
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        print("torchsummary not available. Install with: pip install torchsummary")
        print(f"Model: {model}")
        
        # Count parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")