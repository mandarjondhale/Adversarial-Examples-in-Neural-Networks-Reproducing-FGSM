# Adversarial Examples in Neural Networks — Reproducing FGSM

A hands-on implementation of **adversarial attacks and adversarial training** on a CNN trained on MNIST, based on the seminal paper:

> *Explaining and Harnessing Adversarial Examples* — Goodfellow et al., 2015

---

## Overview

This project demonstrates how small, imperceptible perturbations — called **adversarial examples** — can fool a well-trained neural network into misclassifying images. It then shows how **adversarial training** can harden the model against such attacks.

**Three key stages:**
1. Train a clean CNN on MNIST
2. Attack it using FGSM at various epsilon strengths
3. Re-train with adversarial examples and compare robustness

---

## What is FGSM?

**Fast Gradient Sign Method (FGSM)** computes the gradient of the loss with respect to the input image and perturbs it in the direction that maximally increases the loss:

```
x_adv = x + ε · sign(∇ₓ J(θ, x, y))
```

Even tiny `ε` values (e.g., 0.1–0.3) can drop model accuracy from ~99% to near-random.

---

## Project Structure

```
FML project/
├── FGSM.ipynb                   # Main notebook: training, attack, adversarial training
├── mnist_cnn.pth                # Saved clean-trained CNN weights
├── mnist_cnn_adv_trained.pth    # Saved adversarially-trained CNN weights
├── DATA/                        # MNIST dataset (auto-downloaded)
└── projenv/                     # Python virtual environment
```

---

## Model Architecture — SimpleCNN

A lightweight CNN trained on MNIST (28×28 grayscale):

| Layer | Details |
|-------|---------|
| Conv1 | 1 → 32 filters, 3×3, ReLU |
| MaxPool | 2×2 |
| Conv2 | 32 → 64 filters, 3×3, ReLU |
| MaxPool | 2×2 |
| FC1 | 1600 → 128, ReLU |
| FC2 | 128 → 10 (logits) |

---

## Results

### FGSM Attack (Clean-trained model)

| Epsilon (ε) | Adversarial Accuracy |
|------------|----------------------|
| 0.00 | ~99% |
| 0.05 | drops significantly |
| 0.10 | drops further |
| 0.20 | near collapse |
| 0.30 | near-random |

### Adversarial Training vs Clean Training (ε = 0.2)

| Model | Clean Accuracy | Adv Accuracy (ε=0.2) |
|-------|---------------|----------------------|
| Clean-trained | ~99% | very low |
| Adv-trained | slightly lower | significantly higher |

> Adversarial training uses a **mixed loss**: `0.5 × L(clean) + 0.5 × L(adversarial)`

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| PyTorch | CNN training, FGSM implementation |
| torchvision | MNIST dataset & transforms |
| Matplotlib | Loss/accuracy curves, attack visualizations |
| Jupyter Notebook | Interactive experimentation |

---

## Setup & Usage

```bash
# Clone / navigate to project
cd "FML project"

# Activate virtual environment
projenv\Scripts\activate    # Windows

# Install dependencies
pip install torch torchvision matplotlib jupyter

# Launch notebook
jupyter notebook FGSM.ipynb
```

---

## Key Concepts

- **Adversarial Examples** — inputs crafted to fool ML models
- **FGSM** — single-step gradient-based attack (Goodfellow et al., 2015)
- **Adversarial Training** — augmenting training data with adversarial examples to improve robustness
- **Epsilon (ε)** — controls perturbation strength; higher ε = stronger attack

---

## Reference

Goodfellow, I., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR 2015.
[arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572)

---

## Course Context

**CS 725 — Foundations of Machine Learning**, IIT Bombay (Sem 1)
