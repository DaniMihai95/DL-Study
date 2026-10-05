# Deep Learning Study Guide

A structured, hands-on study guide for a university Deep Learning course.
Covers six topics in course order, with one focused notebook per concept.

Each notebook follows the same structure:
- **Where we are** - placement in the course arc
- **Theory recap** - key formulas and intuitions
- **Your options** - "at this step you can use A, B, or C" with a trade-off table
- **Your turn** - `### YOUR CODE HERE ###` blocks with guided hints
- **Side-by-side comparison** - runs every option and plots results together
- **Exam corner** - typical exam questions with hidden answers

---

## Topics

### 01 - Perceptron

| Notebook | What you build |
|---|---|
| [01a - Neuron Math](01_Perceptron/01a_Neuron_Math.ipynb) | Dot product, bias, activation - three ways to implement a single neuron |
| [01b - Perceptron Training](01_Perceptron/01b_Perceptron_Training.ipynb) | Weight update rule, convergence, learning rate comparison, decision boundary |

### 02 - MLP and Backpropagation

| Notebook | What you build |
|---|---|
| [02a - Forward Pass](02_MLP_Backprop/02a_Forward_Pass.ipynb) | Layer-by-layer computation, shape tracking, NumPy vs Keras comparison |
| [02b - Backpropagation](02_MLP_Backprop/02b_Backpropagation.ipynb) | Chain rule by hand, gradient tape, vanishing gradient demo, numerical gradient check |
| [02c - Activation Functions](02_MLP_Backprop/02c_Activation_Functions.ipynb) | Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Swish, Softmax - all implemented and compared |

### 03 - Convolutional Neural Networks

| Notebook | What you build |
|---|---|
| [03a - Convolution Operation](03_CNN/03a_Convolution_Op.ipynb) | 2D conv from scratch, valid vs same padding, strides, output size formulas |
| [03b - Pooling Types](03_CNN/03b_Pooling_Types.ipynb) | MaxPool, AvgPool, GlobalAvgPool, GlobalMaxPool, stride-conv - all compared |
| [03c - CNN Architectures](03_CNN/03c_CNN_Architectures.ipynb) | LeNet-style vs VGG-style vs ResNet block - trained and compared on CIFAR-10 |
| [03d - Transfer Learning](03_CNN/03d_Transfer_Learning.ipynb) | Frozen base vs fine-tuning vs full fine-tuning vs from scratch - all four strategies |

### 04 - Optimizers and Regularization

| Notebook | What you build |
|---|---|
| [04a - Optimizers](04_Optimizers_Regularization/04a_Optimizers.ipynb) | SGD, Momentum, Nesterov, RMSProp, Adam, AdamW - from scratch and compared on loss curves |
| [04b - Regularization](04_Optimizers_Regularization/04b_Regularization.ipynb) | L1, L2, Dropout, BatchNorm, EarlyStopping, combined - all on the same overfit model |

### 05 - Recurrent Neural Networks

| Notebook | What you build |
|---|---|
| [05a - Vanilla RNN](05_RNN/05a_Vanilla_RNN.ipynb) | RNN cell from scratch, return_sequences explained, vanishing gradient on long sequences |
| [05b - LSTM vs GRU](05_RNN/05b_LSTM_vs_GRU.ipynb) | All gates implemented from scratch, side-by-side accuracy and parameter count |
| [05c - Bidirectional and Stacked](05_RNN/05c_Bidirectional_Stacked.ipynb) | Unidirectional vs Bidirectional vs Stacked vs Stacked-Bidirectional |

### 06 - Attention and Transformers

| Notebook | What you build |
|---|---|
| [06a - Attention Mechanism](06_Attention_Transformers/06a_Attention_Mechanism.ipynb) | Additive, multiplicative, and scaled dot-product attention - all implemented and visualised |
| [06b - Multi-Head Attention](06_Attention_Transformers/06b_MultiHead_Attention.ipynb) | Custom multi-head attention layer, head visualisation, comparison to Keras built-in |
| [06c - Transformer Block](06_Attention_Transformers/06c_Transformer_Block.ipynb) | Positional encoding options, full encoder block, text classifier, full course cheat sheet |

---

## Setup

Uses the virtual environment from the parent course folder.

```bash
# From inside this repo folder
../DL/.venv/bin/jupyter notebook
```

Or create a fresh venv here:

```bash
python3 -m venv .venv
.venv/bin/pip install "tensorflow-cpu==2.20.*" scikit-learn matplotlib seaborn pandas notebook
```

Requirements: Python 3.10+, TensorFlow 2.20, Keras 3. CPU-only. No GPU needed.

---

## Structure

```
.
+-- 01_Perceptron/
+-- 02_MLP_Backprop/
+-- 03_CNN/
+-- 04_Optimizers_Regularization/
+-- 05_RNN/
+-- 06_Attention_Transformers/
+-- README.md
```

---

## Notes

- All notebooks run on CPU. No GPU required.
- Datasets used: MNIST, CIFAR-10, IMDB (all downloaded automatically by Keras/TensorFlow).
- The RNN notebooks can also use `Epileptic_Seizure_Recognition.csv` if available locally.
- `06c` ends with a full course cheat sheet covering every topic from Perceptron to Transformers.
