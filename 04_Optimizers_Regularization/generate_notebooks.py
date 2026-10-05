import json
import os

def get_notebook_04a():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Optimizers\n",
                "**Course:** Deep Learning  \n",
                "**Lecture:** 4 — Optimizers and Regularization  \n",
                "**Part:** a — Optimizers\n",
                "\n",
                "---\n",
                "\n",
                "> **Where we are:** Now that we know how to construct networks, we need to know how to efficiently update their weights to minimize the loss function.\n",
                "\n",
                "**What you'll learn:**\n",
                "- Standard gradient descent and its limitations\n",
                "- Momentum and Nesterov accelerated gradient\n",
                "- Adaptive learning rate methods like RMSProp and Adam\n",
                "- How to use and compare them in TensorFlow\n",
                "\n",
                "**Exam relevance:** ⭐⭐⭐ (Very important! You will likely be asked to explain Adam or compare it to SGD with momentum.)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Imports ────────────────────────────────────────────────────────────────────\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import tensorflow as tf\n",
                "\n",
                "# Confirm device\n",
                "print(f\"TensorFlow {tf.__version__} | Devices: {[d.name for d in tf.config.list_physical_devices()]}\")\n",
                "\n",
                "# Reproducibility\n",
                "np.random.seed(42)\n",
                "tf.random.set_seed(42)\n",
                "\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 5)\n",
                "plt.rcParams['font.size'] = 12\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📖 Theory: Optimization Algorithms\n",
                "When training a neural network, we use gradient descent to update weights $w$. \n",
                "\n",
                "- **SGD (Vanilla):** $w \\leftarrow w - \\eta \\nabla L$. Slow, can get stuck in local minima or saddle points.\n",
                "- **SGD + Momentum:** Accumulates a velocity vector to build up speed in consistent directions.\n",
                "  $v \\leftarrow \\beta v + \\nabla L$\n",
                "  $w \\leftarrow w - \\eta v$\n",
                "- **RMSProp:** Divides learning rate by a moving average of squared gradients to adapt per-weight.\n",
                "- **Adam:** Combines Momentum (first moment $m$) and RMSProp (second moment $v$). The gold standard.\n",
                "  $m \\leftarrow \\beta_1 m + (1-\\beta_1) \\nabla L$\n",
                "  $v \\leftarrow \\beta_2 v + (1-\\beta_2) \\nabla L^2$\n",
                "  Bias correction is applied in the first few steps.\n",
                "- **AdamW:** Adam with decoupled weight decay.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔀 Your Options\n",
                "At this step, you can choose between several optimizers. Here is a comparison:\n",
                "\n",
                "| Option | Pros | Cons | When to use |\n",
                "|---|---|---|---|\n",
                "| **SGD** | Simple, theoretically sound | Very slow, needs careful tuning | Rarely used vanilla |\n",
                "| **SGD+Momentum** | Much faster convergence | Hyperparameter $\\beta$ needs tuning | Often for vision tasks (e.g., ResNets) |\n",
                "| **RMSProp** | Adapts LR per weight | Can stall if LR drops too fast | RNNs, sequential data |\n",
                "| **Adam** | Fast, default choice, adaptive | Might overfit compared to SGD | Almost everywhere as a default |\n",
                "| **AdamW** | Better generalization than Adam | Slightly more tuning for weight decay | Modern transformers, heavy vision models |"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Part 1: Visualizing Optimizers on a 2D Loss Surface (Rosenbrock)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def rosenbrock(x, y):\n",
                "    return (1 - x)**2 + 100 * (y - x**2)**2\n",
                "\n",
                "def rosenbrock_grad(x, y):\n",
                "    dx = -2 * (1 - x) - 400 * x * (y - x**2)\n",
                "    dy = 200 * (y - x**2)\n",
                "    return np.array([dx, dy])\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### YOUR CODE HERE ###\n",
                "# Implement standard SGD update rule\n",
                "def sgd_update(w, grad, lr):\n",
                "    # w_new = ...\n",
                "    pass\n",
                "\n",
                "# Implement SGD + Momentum\n",
                "def momentum_update(w, grad, lr, v, beta):\n",
                "    # v_new = ...\n",
                "    # w_new = ...\n",
                "    pass\n",
                "\n",
                "# Implement Adam with bias correction\n",
                "def adam_update(w, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):\n",
                "    # m_new = ...\n",
                "    # v_new = ...\n",
                "    # m_hat = ... (bias correction)\n",
                "    # v_hat = ... (bias correction)\n",
                "    # w_new = ...\n",
                "    pass\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### ✅ SOLUTION ###\n",
                "def sgd_update(w, grad, lr):\n",
                "    return w - lr * grad\n",
                "\n",
                "def momentum_update(w, grad, lr, v, beta):\n",
                "    v_new = beta * v + grad\n",
                "    w_new = w - lr * v_new\n",
                "    return w_new, v_new\n",
                "\n",
                "def adam_update(w, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):\n",
                "    m_new = beta1 * m + (1 - beta1) * grad\n",
                "    v_new = beta2 * v + (1 - beta2) * (grad ** 2)\n",
                "    \n",
                "    # Bias correction\n",
                "    m_hat = m_new / (1 - beta1 ** t)\n",
                "    v_hat = v_new / (1 - beta2 ** t)\n",
                "    \n",
                "    w_new = w - lr * m_hat / (np.sqrt(v_hat) + eps)\n",
                "    return w_new, m_new, v_new\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### 🔬 Comparison Cell: Trajectories\n",
                "np.random.seed(42)\n",
                "start_w = np.array([-1.0, -1.0])\n",
                "iterations = 1000\n",
                "\n",
                "def run_optimizer(opt_name, start_w, iterations):\n",
                "    w = start_w.copy()\n",
                "    path = [w.copy()]\n",
                "    \n",
                "    # States\n",
                "    v_mom = np.zeros_like(w)\n",
                "    m_adam, v_adam = np.zeros_like(w), np.zeros_like(w)\n",
                "    \n",
                "    for t in range(1, iterations + 1):\n",
                "        g = rosenbrock_grad(w[0], w[1])\n",
                "        \n",
                "        if opt_name == 'SGD':\n",
                "            w = sgd_update(w, g, lr=0.001)\n",
                "        elif opt_name == 'Momentum':\n",
                "            w, v_mom = momentum_update(w, g, lr=0.001, v=v_mom, beta=0.9)\n",
                "        elif opt_name == 'Adam':\n",
                "            w, m_adam, v_adam = adam_update(w, g, lr=0.1, m=m_adam, v=v_adam, t=t)\n",
                "            \n",
                "        path.append(w.copy())\n",
                "    return np.array(path)\n",
                "\n",
                "path_sgd = run_optimizer('SGD', start_w, iterations)\n",
                "path_mom = run_optimizer('Momentum', start_w, iterations)\n",
                "path_adam = run_optimizer('Adam', start_w, iterations)\n",
                "\n",
                "# Plot\n",
                "X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-1, 3, 100))\n",
                "Z = rosenbrock(X, Y)\n",
                "\n",
                "plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')\n",
                "plt.plot(path_sgd[:,0], path_sgd[:,1], label='SGD', alpha=0.8)\n",
                "plt.plot(path_mom[:,0], path_mom[:,1], label='Momentum', alpha=0.8)\n",
                "plt.plot(path_adam[:,0], path_adam[:,1], label='Adam', alpha=0.8)\n",
                "plt.plot(1, 1, 'r*', markersize=15, label='Global Min')\n",
                "plt.legend()\n",
                "plt.title('Optimizer Trajectories on Rosenbrock Function')\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Part 2: Training a MLP on MNIST"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load MNIST\n",
                "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()\n",
                "x_train, x_test = x_train / 255.0, x_test / 255.0\n",
                "\n",
                "def build_model():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Flatten(input_shape=(28, 28)),\n",
                "        tf.keras.layers.Dense(128, activation='relu'),\n",
                "        tf.keras.layers.Dense(10, activation='softmax')\n",
                "    ])\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### YOUR CODE HERE ###\n",
                "# Define the Keras optimizers to compare\n",
                "optimizers_to_test = {\n",
                "    'SGD': None,  # tf.keras.optimizers.SGD(...)\n",
                "    'Momentum': None, # tf.keras.optimizers.SGD(..., momentum=0.9)\n",
                "    'RMSProp': None,\n",
                "    'Adam': None,\n",
                "    'AdamW': None\n",
                "}\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### ✅ SOLUTION ###\n",
                "optimizers_to_test = {\n",
                "    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),\n",
                "    'Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),\n",
                "    'RMSProp': tf.keras.optimizers.RMSprop(learning_rate=0.001),\n",
                "    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),\n",
                "    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)\n",
                "}\n",
                "\n",
                "histories = {}\n",
                "for name, opt in optimizers_to_test.items():\n",
                "    print(f\"Training with {name}...\")\n",
                "    tf.random.set_seed(42)\n",
                "    model = build_model()\n",
                "    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
                "    hist = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)\n",
                "    histories[name] = hist\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### 🔬 Comparison Cell: Loss & Accuracy\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n",
                "\n",
                "for name, hist in histories.items():\n",
                "    ax1.plot(hist.history['loss'], label=name)\n",
                "    ax2.plot(hist.history['val_accuracy'], label=name)\n",
                "\n",
                "ax1.set_title('Training Loss')\n",
                "ax1.set_xlabel('Epoch')\n",
                "ax1.set_ylabel('Loss')\n",
                "ax1.legend()\n",
                "\n",
                "ax2.set_title('Validation Accuracy')\n",
                "ax2.set_xlabel('Epoch')\n",
                "ax2.set_ylabel('Accuracy')\n",
                "ax2.legend()\n",
                "\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎓 Exam Corner\n",
                "\n",
                "### Question 1 (Conceptual)\n",
                "Write the Adam update rule. What are $\\beta_1$ and $\\beta_2$?\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "$m_t = \\beta_1 m_{t-1} + (1-\\beta_1) \\nabla L$\n",
                "$v_t = \\beta_2 v_{t-1} + (1-\\beta_2) \\nabla L^2$\n",
                "$\\hat{m}_t = m_t / (1 - \\beta_1^t)$ (bias correction)\n",
                "$\\hat{v}_t = v_t / (1 - \\beta_2^t)$ (bias correction)\n",
                "$w_{t} = w_{t-1} - \\eta \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}$\n",
                "\n",
                "$\\beta_1$ controls the momentum (decay of first moment). $\\beta_2$ controls RMSProp part (decay of second moment/variance).\n",
                "\n",
                "</details>\n",
                "\n",
                "### Question 2 (Conceptual)\n",
                "What is the difference between SGD with momentum and Adam?\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "SGD with momentum uses a single global learning rate and accumulates past gradients. Adam additionally maintains a moving average of squared gradients (second moment) to adjust the learning rate *per parameter*, scaling down updates for weights with large gradients and scaling up for those with small gradients.\n",
                "\n",
                "</details>\n",
                "\n",
                "### Question 3 (Conceptual)\n",
                "Why does Adam need bias correction in the first few steps?\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "The moving averages $m$ and $v$ are initialized to zeros. In the early steps, they are biased heavily towards zero because the decay rates (e.g. 0.999) keep most of the previous zero value. Bias correction adjusts for this, ensuring the first steps have proper magnitude.\n",
                "\n",
                "</details>"
            ]
        }
    ]
    
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

def get_notebook_04b():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🧠 Regularization\n",
                "**Course:** Deep Learning  \n",
                "**Lecture:** 4 — Optimizers and Regularization  \n",
                "**Part:** b — Regularization\n",
                "\n",
                "---\n",
                "\n",
                "> **Where we are:** We know how to optimize our model. But if it optimizes too well on training data, it overfits. Now we learn to constrain it.\n",
                "\n",
                "**What you'll learn:**\n",
                "- L1 and L2 weight regularization\n",
                "- Dropout and how it randomly drops activations\n",
                "- Batch Normalization and its stabilizing effect\n",
                "- Early Stopping to halt training at the right time\n",
                "\n",
                "**Exam relevance:** ⭐⭐⭐ (Extremely relevant. You must be able to diagnose overfitting and choose the correct mitigation.)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ── Imports ────────────────────────────────────────────────────────────────────\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import tensorflow as tf\n",
                "\n",
                "# Confirm device\n",
                "print(f\"TensorFlow {tf.__version__} | Devices: {[d.name for d in tf.config.list_physical_devices()]}\")\n",
                "\n",
                "# Reproducibility\n",
                "np.random.seed(42)\n",
                "tf.random.set_seed(42)\n",
                "\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "plt.rcParams['figure.figsize'] = (10, 5)\n",
                "plt.rcParams['font.size'] = 12\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📖 Theory: Regularization Techniques\n",
                "When train loss $\\ll$ validation loss, the model is overfitting.\n",
                "\n",
                "- **L1 Regularization:** Adds $\\lambda \\sum |w|$ to the loss. Induces sparsity (many weights become zero).\n",
                "- **L2 Regularization (Weight Decay):** Adds $\\lambda \\sum w^2$ to the loss. Shrinks weights evenly.\n",
                "- **Dropout:** Randomly zeroes out neurons with probability $p$ during training, preventing co-adaptation.\n",
                "- **Batch Normalization:** Normalizes layer inputs to mean 0, variance 1. Technically an optimization aid, but acts as a weak regularizer.\n",
                "- **Early Stopping:** Halts training when validation metric stops improving.\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔀 Your Options\n",
                "At this step, you can choose between several regularization strategies:\n",
                "\n",
                "| Option | Pros | Cons | When to use |\n",
                "|---|---|---|---|\n",
                "| **None** | Baseline | Overfits | Prototyping |\n",
                "| **L1/L2 Weight Decay** | Smooth bounds on weights | Tuning $\\lambda$ required | Almost always (L2) |\n",
                "| **Dropout** | Very effective anti-overfitting | Slower convergence | Dense layers, large models |\n",
                "| **Batch Norm** | Faster training, allows larger LR | Small batch size issues | Deep CNNs/MLPs |\n",
                "| **Early Stopping** | Prevents \"over-training\" | Might stop too early | Always (as a callback) |"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load IMDB data (reduced size to force overfitting)\n",
                "vocab_size = 5000\n",
                "(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)\n",
                "\n",
                "# Pad sequences\n",
                "maxlen = 100\n",
                "x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)\n",
                "x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)\n",
                "\n",
                "# Subset to make it overfit easily\n",
                "x_train, y_train = x_train[:2000], y_train[:2000]\n",
                "x_test, y_test = x_test[:1000], y_test[:1000]\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### YOUR CODE HERE ###\n",
                "# 1. Build unregularized overfit baseline\n",
                "# 2. Build L1 model\n",
                "# 3. Build L2 model\n",
                "# 4. Build Dropout model\n",
                "# 5. Build Batch Norm model\n",
                "# 6. Build Combined model\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### ✅ SOLUTION ###\n",
                "def get_baseline():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, activation='relu'),\n",
                "        tf.keras.layers.Dense(64, activation='relu'),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "def get_l1():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),\n",
                "        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001)),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "def get_l2():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),\n",
                "        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "def get_dropout(rate=0.5):\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, activation='relu'),\n",
                "        tf.keras.layers.Dropout(rate),\n",
                "        tf.keras.layers.Dense(64, activation='relu'),\n",
                "        tf.keras.layers.Dropout(rate),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "def get_batchnorm():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, use_bias=False),\n",
                "        tf.keras.layers.BatchNormalization(),\n",
                "        tf.keras.layers.Activation('relu'),\n",
                "        tf.keras.layers.Dense(64, use_bias=False),\n",
                "        tf.keras.layers.BatchNormalization(),\n",
                "        tf.keras.layers.Activation('relu'),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "def get_combined():\n",
                "    return tf.keras.Sequential([\n",
                "        tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen),\n",
                "        tf.keras.layers.Flatten(),\n",
                "        tf.keras.layers.Dense(64, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(0.001)),\n",
                "        tf.keras.layers.BatchNormalization(),\n",
                "        tf.keras.layers.Activation('relu'),\n",
                "        tf.keras.layers.Dropout(0.3),\n",
                "        tf.keras.layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "\n",
                "models = {\n",
                "    'Baseline': get_baseline(),\n",
                "    'L1': get_l1(),\n",
                "    'L2': get_l2(),\n",
                "    'Dropout': get_dropout(0.5),\n",
                "    'BatchNorm': get_batchnorm(),\n",
                "    'Combined': get_combined()\n",
                "}\n",
                "\n",
                "histories = {}\n",
                "for name, model in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
                "    \n",
                "    callbacks = []\n",
                "    if name == 'Combined':\n",
                "        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))\n",
                "        \n",
                "    histories[name] = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=callbacks, verbose=0)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "### 🔬 Comparison Cell: Learning Curves & Weight Distribution\n",
                "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for i, (name, hist) in enumerate(histories.items()):\n",
                "    axes[i].plot(hist.history['loss'], label='Train Loss')\n",
                "    axes[i].plot(hist.history['val_loss'], label='Val Loss')\n",
                "    axes[i].set_title(f\"{name}\")\n",
                "    axes[i].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Weight distribution comparison\n",
                "w_base = models['Baseline'].layers[2].get_weights()[0].flatten()\n",
                "w_l1 = models['L1'].layers[2].get_weights()[0].flatten()\n",
                "w_l2 = models['L2'].layers[2].get_weights()[0].flatten()\n",
                "\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.hist(w_base, bins=50, alpha=0.5, label='Baseline', density=True)\n",
                "plt.hist(w_l1, bins=50, alpha=0.5, label='L1 (Sparse)', density=True)\n",
                "plt.hist(w_l2, bins=50, alpha=0.5, label='L2 (Shrunk)', density=True)\n",
                "plt.title(\"Weight Distribution Comparison\")\n",
                "plt.legend()\n",
                "plt.xlim(-0.2, 0.2)\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🎓 Exam Corner\n",
                "\n",
                "### Question 1 (Conceptual)\n",
                "What is the difference between L1 and L2 regularization? Which produces sparser weights?\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "L1 penalizes the absolute value of weights, pulling many exactly to zero, thus acting as feature selection (sparsity). L2 penalizes squared weights, which shrinks large weights heavily but rarely forces them entirely to zero. L1 produces sparser weights.\n",
                "\n",
                "</details>\n",
                "\n",
                "### Question 2 (Conceptual)\n",
                "Where exactly is BatchNormalization applied in a network? (before or after activation?)\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "The original paper recommends applying it *before* the non-linear activation (Dense -> BatchNorm -> ReLU). However, empirically, some modern architectures apply it *after* the activation. Before activation is the \"textbook\" answer.\n",
                "\n",
                "</details>\n",
                "\n",
                "### Question 3 (Practical)\n",
                "Explain what Dropout does during training vs during inference.\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "During training, it randomly zeroes out neurons with probability $p$. To keep the expected sum of activations the same, the active neurons are scaled by $1 / (1-p)$. During inference, Dropout is turned off (all neurons are active), and no scaling is applied (since it was handled at training time).\n",
                "\n",
                "</details>\n",
                "\n",
                "### Question 4 (Practical)\n",
                "What does `restore_best_weights=True` do in EarlyStopping?\n",
                "\n",
                "<details><summary>💡 Answer</summary>\n",
                "\n",
                "When early stopping halts training because the validation metric hasn't improved for `patience` epochs, the model is at its final state. `restore_best_weights=True` reverts the model's weights to the state they were in at the epoch with the best validation score, rather than the final overfitted epoch.\n",
                "\n",
                "</details>"
            ]
        }
    ]
    
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }

if __name__ == '__main__':
    with open('/home/dani/Documents/DL/04_Study/04_Optimizers_Regularization/04a_Optimizers.ipynb', 'w') as f:
        json.dump(get_notebook_04a(), f, indent=1)
    with open('/home/dani/Documents/DL/04_Study/04_Optimizers_Regularization/04b_Regularization.ipynb', 'w') as f:
        json.dump(get_notebook_04b(), f, indent=1)
