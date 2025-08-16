# GenGenerative-Adversarial-Networks
🧵 Vanilla GAN on FashionMNIST

This repository contains an implementation of a Vanilla Generative Adversarial Network (GAN) trained on the FashionMNIST dataset. The project is inspired by the seminal work Generative Adversarial Nets by Ian Goodfellow et al. (2014), where a generator and discriminator compete in a minimax game to improve generative modeling.

🚀 Project Overview

The goal of this project is to build and train a Vanilla GAN (i.e., without architectural optimizations like CNNs or DCGAN improvements) to generate realistic fashion item images such as shirts, trousers, shoes, and bags.

GANs work by training two neural networks simultaneously:

Generator (G): Takes random noise as input and generates fake images resembling the training data.

Discriminator (D): Classifies inputs as real (from dataset) or fake (from generator).

During training, G tries to fool D, while D learns to distinguish real vs. fake. This adversarial game improves both models until G produces realistic outputs.

📂 Repository Structure

Vanilla_GAN_FashionMNIST.ipynb → Jupyter notebook containing the full GAN implementation, training loop, and visualization of results.

GAN_Research_Paper.pdf → Original research paper by Goodfellow et al., serving as the theoretical foundation.

/results/ (optional) → Stores generated images across epochs for qualitative comparison.

🧑‍💻 Implementation Details

Framework: Python, PyTorch / TensorFlow (depending on your codebase).

Dataset: FashionMNIST, consisting of 60,000 training and 10,000 test grayscale images (28×28).

Model Architecture:

Generator: Fully connected layers with non-linear activations (ReLU/LeakyReLU).

Discriminator: Fully connected layers with sigmoid activation at the output.

Loss Function: Binary Cross-Entropy Loss (BCE).

Optimization: Adam optimizer with tuned hyperparameters (learning rate, β1).

Training Strategy: Alternate updates between D and G, as outlined in the original paper
.

📊 Results

Generated samples begin as noisy blobs but gradually improve to resemble fashion items after sufficient epochs.

Discriminator accuracy fluctuates around 50% at convergence, indicating that the generator produces realistic outputs.

Example outputs at different training stages:
(Insert sample image grid here)

🔬 Key Learnings

Vanilla GANs are unstable: Without architectural improvements, training is prone to mode collapse and oscillations.

Hyperparameters matter: Learning rate, batch size, and optimizer β values significantly affect stability.

Visualization is crucial: Tracking losses and generated samples helps diagnose GAN behavior.

📖 References

Goodfellow, Ian, et al. Generative Adversarial Nets. NeurIPS 2014. 

FashionMNIST Dataset: Zalando Research

✅ Future Work

Implement DCGAN with convolutional layers for improved image quality.

Explore conditional GANs (cGANs) to generate class-specific fashion items.

Apply GAN evaluation metrics (e.g., Inception Score, FID) for quantitative benchmarking.
