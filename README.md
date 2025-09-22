# Adversarial Robustness on CIFAR-10

A comprehensive implementation of adversarial attacks and defenses for deep neural networks on the CIFAR-10 dataset. This project explores the vulnerability of convolutional neural networks to adversarial examples and implements various techniques to improve model robustness.

## 🎯 Project Overview

This project demonstrates the fragility of deep learning models to carefully crafted adversarial perturbations and implements state-of-the-art defense mechanisms. Using the CIFAR-10 dataset, we:

- **Fine-tune pretrained ResNet-18** models for image classification
- **Generate adversarial examples** using FGSM and PGD attacks
- **Evaluate model robustness** against various attack strategies
- **Implement adversarial training** to improve model resilience
- **Compare clean vs. adversarial accuracy** across different model configurations

## 🏗️ Repository Structure

```
Project-DL-Opti/
├── data/                           # Dataset storage
│   ├── cifar-10-python.tar.gz     # CIFAR-10 dataset archive
│   └── cifar-10-batches-py/        # Extracted CIFAR-10 data
├── experiments/                    # Completed experiments
│   └── adversarial_robustness_completed.ipynb  # Full implementation
├── notebooks/                      # Tutorial and development notebooks
│   └── adversarial_attacks_tutorial.ipynb      # Step-by-step tutorial
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── adversarial_attacks.py    # Attack implementations (FGSM, PGD)
│   ├── data_utils.py            # Dataset loading and visualization
│   └── model_utils.py           # Model training and evaluation
├── results/                      # Experimental results and plots
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Key Features

### Adversarial Attacks

- **Fast Gradient Sign Method (FGSM)**: Single-step gradient-based attack
- **Projected Gradient Descent (PGD)**: Multi-step iterative attack
- **Customizable attack parameters**: Epsilon values, step sizes, iterations

### Model Architectures

- **ResNet-18**: Pre-trained on ImageNet, fine-tuned for CIFAR-10
- **Custom modifications**: Adapted final layer for 10-class classification
- **Model evaluation**: Comprehensive accuracy metrics and robustness analysis

### Defense Mechanisms

- **Adversarial Training**: Training with adversarial examples in the loop
- **Data Augmentation**: Standard techniques to improve generalization
- **Robust Optimization**: SGD with carefully tuned hyperparameters

### Visualization & Analysis

- **Side-by-side comparisons**: Clean vs. adversarial examples
- **Attack success rates**: Detailed robustness evaluation
- **Training curves**: Loss and accuracy progression
- **Gradient analysis**: Input gradient visualization

## 📦 Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/pottierc/Project-DL-Opti.git
   cd Project-DL-Opti
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download CIFAR-10** (automatic on first run):
   The dataset will be automatically downloaded when running the notebooks.

## 🎮 Usage

### Quick Start

1. **Open the tutorial notebook**:

   ```bash
   jupyter notebook notebooks/adversarial_attacks_tutorial.ipynb
   ```

2. **Follow the step-by-step implementation** covering:
   - CIFAR-10 data loading and preprocessing
   - ResNet-18 model adaptation
   - Adversarial attack implementation
   - Model robustness evaluation

### Advanced Usage

For the complete implementation, explore:

```bash
jupyter notebook experiments/adversarial_robustness_completed.ipynb
```

### Using the Python Modules

```python
from src import get_cifar10_loaders, get_resnet18_cifar10, fgsm_attack, evaluate_robustness

# Load data
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# Initialize model
model = get_resnet18_cifar10(pretrained=True)

# Generate adversarial examples
adv_images = fgsm_attack(model, images, labels, epsilon=0.03)

# Evaluate robustness
clean_acc, adv_acc = evaluate_robustness(model, test_loader, fgsm_attack, device)
```

## 🧪 Experimental Results

### Model Performance

- **Clean Accuracy**: ~85-90% on CIFAR-10 test set
- **FGSM Attack (ε=0.03)**: ~20-40% accuracy drop
- **PGD Attack (ε=0.03)**: ~30-50% accuracy drop
- **Adversarial Training**: Improved robustness with ~10-15% clean accuracy trade-off

### Key Insights

1. **Transfer Learning**: Pre-trained ImageNet models require adaptation for CIFAR-10
2. **Attack Strength**: Higher epsilon values lead to more successful attacks
3. **Defense Trade-offs**: Adversarial training improves robustness at the cost of clean accuracy
4. **Iterative Attacks**: PGD consistently outperforms FGSM in attack success

## 📚 Technical Background

### Adversarial Examples

Adversarial examples are inputs to machine learning models that are intentionally designed to cause the model to make mistakes. They are created by adding small, often imperceptible perturbations to legitimate inputs.

### Mathematical Formulation

- **Clean Training**: `min_θ E[L(f_θ(x), y)]`
- **Adversarial Training**: `min_θ E[max_{||δ||≤ε} L(f_θ(x+δ), y)]`
- **FGSM Attack**: `x_adv = x + ε × sign(∇_x L(f_θ(x), y))`
- **PGD Attack**: Iterative application of FGSM with projection

## 🔬 Research Context

This project implements concepts from several seminal papers in adversarial machine learning:

- **Goodfellow et al. (2014)**: "Explaining and Harnessing Adversarial Examples"
- **Madry et al. (2018)**: "Towards Deep Learning Models Resistant to Adversarial Attacks"
- **Szegedy et al. (2013)**: "Intriguing properties of neural networks"

## 🎓 Educational Value

### For Students

- Hands-on experience with adversarial machine learning
- Understanding of model vulnerabilities and defenses
- Practical implementation of research concepts
- Visualization of attack mechanisms

### For Interviews

- Demonstrates knowledge of cutting-edge ML security
- Shows practical implementation skills
- Highlights understanding of model robustness
- Exhibits ability to work with real datasets and pretrained models

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, torchvision
- **Data Science**: NumPy, Matplotlib
- **Development**: Jupyter Notebooks, Python 3.7+
- **Visualization**: Custom plotting utilities
- **Dataset**: CIFAR-10 (60,000 32×32 color images)

## 📈 Future Extensions

- Implementation of additional attacks (C&W, AutoAttack)
- Certified defense mechanisms
- Ensemble methods for improved robustness
- Extension to other datasets (ImageNet, CIFAR-100)
- Real-time adversarial detection

## 🤝 Contributing

This is an academic project developed for the Deep Learning & Optimization course. For questions or suggestions, please contact the authors.

## 📄 License

This project is developed for educational purposes as part of academic coursework.

---

**Authors**: Victor Soto, Clément Pottier  
**Course**: Deep Learning & Optimization  
**Institution**: HEC Paris  
**Academic Year**: 2025
