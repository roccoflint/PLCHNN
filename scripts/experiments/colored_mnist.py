import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np

from core.network import HebbianNetwork, to_device
from core.utils import (
    load_mnist, compute_confusion_matrix, plot_confusion_matrix,
    plot_association_matrix, plot_reconstructions,
    plot_reconstruction_similarity_matrix, cosine_similarity,
    visualize_weights, multichannel_to_rgb, COLORS, COLOR_LABELS
)
import scipy.io
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from datetime import datetime

# Define RGB values for each color
COLORS = {
    3: np.array([  # RGB values for 3 colors
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 0.8, 0.0],  # Green
    ]),
    5: np.array([  # RGB values for 5 colors
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 0.8, 0.0],  # Green
        [0.8, 0.0, 0.8],  # Purple
        [1.0, 0.6, 0.0],  # Orange
    ])
}

def multichannel_to_rgb(data, n_colors):
    """Convert multi-channel representation to RGB image.
    
    Args:
        data: Array of shape (784 * (1 + n_colors),) or (batch_size, 784 * (1 + n_colors))
        n_colors: Number of colors used in the dataset
        
    Returns:
        Array of shape (28, 28, 3) or (batch_size, 28, 28, 3) containing RGB values
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    batch_size = data.shape[0]
    xp = cp if isinstance(data, cp.ndarray) else np
    colors = to_device(COLORS[n_colors], data.device if hasattr(data, 'device') else 'cpu')
    
    # Extract intensity and color channels
    intensity = data[:, :784].reshape(batch_size, 28, 28)
    color_channels = [
        data[:, (784 * (i + 1)):(784 * (i + 2))].reshape(batch_size, 28, 28)
        for i in range(n_colors)
    ]
    
    # Initialize RGB output
    rgb = xp.zeros((batch_size, 28, 28, 3))
    
    # Combine channels using color mapping
    for i, color_channel in enumerate(color_channels):
        for j in range(3):  # RGB channels
            rgb[:, :, :, j] += color_channel * colors[i, j]
    
    # Scale by intensity
    rgb = rgb * intensity.reshape(batch_size, 28, 28, 1)
    
    # Ensure values are in [0, 1]
    rgb = xp.clip(rgb, 0, 1)
    
    return rgb[0] if data.shape[0] == 1 else rgb

def visualize_colored_digits(data, labels, n_colors, n_samples=10):
    """Visualize colored MNIST digits.
    
    Args:
        data: Array of colored MNIST data
        labels: Array of [digit, color] labels
        n_colors: Number of colors used
        n_samples: Number of samples to visualize
    """
    # Convert to CPU numpy arrays if needed
    if HAS_CUDA and isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
        labels = cp.asnumpy(labels)
    
    # Select random indices
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    
    # Create subplot grid
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row, col = i // n_cols, i % n_cols
        rgb = multichannel_to_rgb(data[idx], n_colors)
        axes[row, col].imshow(rgb)
        # Use new labeling scheme: color letter + digit
        digit = labels[idx, 0]
        color = labels[idx, 1]
        label = f"{COLOR_LABELS[color]}{digit}"
        axes[row, col].set_title(label)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def labels_to_signific(labels, n_colors):
    """Convert digit-color labels to n-hot signific vectors.
    
    Args:
        labels: Array of shape (n_samples, 2) with [digit, color] labels
        n_colors: Number of colors used
        
    Returns:
        Array of shape (n_samples, 10 + n_colors) with 2-hot encoding
        First 10 positions for digits, next n_colors positions for colors
    """
    xp = cp if isinstance(labels, cp.ndarray) else np
    n_samples = len(labels)
    signific_size = 10 + n_colors
    signific = xp.zeros((n_samples, signific_size))
    
    # Set digit positions (first 10)
    digit_idx = xp.arange(n_samples)
    signific[digit_idx, labels[:, 0]] = 1
    
    # Set color positions (last n_colors)
    signific[digit_idx, 10 + labels[:, 1]] = 1
    
    return signific

def generate_colored_mnist(n_colors=3, device='cpu', include_signific=True, holdout_pairs=None, train_ratio=0.8):
    """Generate colored MNIST dataset.
    
    Args:
        n_colors (int): Number of colors to use
        device (str): Device to generate data on ('cpu' or 'gpu')
        include_signific (bool): Whether to include signific vectors in labels
        holdout_pairs (list): List of (digit, color) tuples to hold out from training
        train_ratio (float): Ratio of combinations to keep in training set
        
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels, holdout_pairs)
    """
    # Load MNIST data
    mnist_data, mnist_labels, mnist_test_data, mnist_test_labels = load_mnist()
    xp = cp if device == 'gpu' and HAS_CUDA else np
    
    # Calculate number of combinations to hold out (round up to ensure we hold out enough)
    total_combinations = 10 * n_colors
    n_holdout = int(np.ceil(total_combinations * (1 - train_ratio)))
    
    # Generate holdout pairs if none specified
    if holdout_pairs is None:
        # Create list of available digits starting from 9
        available_digits = list(range(9, -1, -1))  # [9, 8, 7, ..., 0]
        
        # Generate holdout pairs by alternating through colors
        holdout_pairs = []
        current_color = 0
        while len(holdout_pairs) < n_holdout and available_digits:
            digit = available_digits.pop(0)  # Take from the start (highest digits)
            holdout_pairs.append((digit, current_color))
            current_color = (current_color + 1) % n_colors  # Cycle through colors
        
        # Sort by digit then color for consistent order
        holdout_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)  # Sort descending by digit
    
    print(f"Holding out {len(holdout_pairs)} combinations: {', '.join([f'{COLOR_LABELS[c]}{d}' for d, c in holdout_pairs])}")
    
    def colorize_data(data, labels, is_training=True):
        n_samples = len(data)
        n_features = 784 * (1 + n_colors)
        colored_data = xp.zeros((n_samples, n_features))
        color_labels = xp.zeros(n_samples, dtype=xp.int64)
        
        # Copy intensity channel
        colored_data[:, :784] = data
        
        # Assign colors ensuring even distribution and respecting holdout pairs
        for digit in range(10):
            digit_mask = labels == digit
            n_digit_samples = int(xp.sum(digit_mask).item())
            
            # Determine available colors for this digit
            if is_training:
                available_colors = [c for c in range(n_colors) if (digit, c) not in holdout_pairs]
            else:
                available_colors = list(range(n_colors))  # All colors available for test
            
            # Create color assignments
            if len(available_colors) > 0:  # Skip if no colors available
                colors = np.array([available_colors[i % len(available_colors)] for i in range(n_digit_samples)])
                np.random.shuffle(colors)
                if HAS_CUDA and isinstance(data, cp.ndarray):
                    colors = cp.array(colors)
                
                # Assign colors to samples with this digit
                color_labels[digit_mask] = colors
                
                # Set color channels
                for color in range(n_colors):
                    color_mask = color_labels == color
                    combined_mask = digit_mask & color_mask
                    start_idx = 784 * (color + 1)
                    end_idx = start_idx + 784
                    colored_data[combined_mask, start_idx:end_idx] = data[combined_mask]
        
        # Stack labels
        labels = xp.stack([labels, color_labels], axis=1)
        
        # Add signific vectors if requested
        if include_signific:
            signific = labels_to_signific(labels, n_colors)
            labels = xp.concatenate([labels, signific], axis=1)
        
        return colored_data, labels
    
    # Generate colored datasets
    train_data, train_labels = colorize_data(mnist_data, mnist_labels, is_training=True)
    test_data, test_labels = colorize_data(mnist_test_data, mnist_test_labels, is_training=False)
    
    return train_data, train_labels, test_data, test_labels, holdout_pairs

def generate_holdout_splits(data, labels, holdout_pairs, train_ratio=0.8):
    """Generate train/test splits holding out specific digit-color pairs.
    
    Args:
        data: Array of shape (n_samples, n_features)
        labels: Array of shape (n_samples, 2) with [digit, color] labels
        holdout_pairs: List of (digit, color) tuples to hold out
        train_ratio: Ratio of non-holdout data to use for training
        
    Returns:
        tuple: (train_data, train_labels, test_data, test_labels)
    """
    xp = cp if isinstance(data, cp.ndarray) else np
    
    # Create mask for holdout pairs
    holdout_mask = xp.zeros(len(data), dtype=bool)
    for digit, color in holdout_pairs:
        pair_mask = (labels[:, 0] == digit) & (labels[:, 1] == color)
        holdout_mask = holdout_mask | pair_mask
    
    # Split non-holdout data into train/test
    non_holdout_mask = ~holdout_mask
    non_holdout_indices = xp.where(non_holdout_mask)[0]
    xp.random.shuffle(non_holdout_indices)
    split_idx = int(len(non_holdout_indices) * train_ratio)
    
    train_indices = non_holdout_indices[:split_idx]
    test_indices = xp.concatenate([
        non_holdout_indices[split_idx:],
        xp.where(holdout_mask)[0]
    ])
    
    return (
        data[train_indices], labels[train_indices],
        data[test_indices], labels[test_indices]
    )

def save_colored_mnist(save_dir, train_data, train_labels, test_data, test_labels):
    """Save colored MNIST dataset."""
    os.makedirs(save_dir, exist_ok=True)
    xp = cp if isinstance(train_data, cp.ndarray) else np
    
    # Move to CPU if needed
    if isinstance(train_data, cp.ndarray):
        train_data = cp.asnumpy(train_data)
        train_labels = cp.asnumpy(train_labels)
        test_data = cp.asnumpy(test_data)
        test_labels = cp.asnumpy(test_labels)
    
    np.savez(
        os.path.join(save_dir, 'colored_mnist.npz'),
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels
    )

def load_colored_mnist(load_dir, device='cpu'):
    """Load colored MNIST dataset."""
    data = np.load(os.path.join(load_dir, 'colored_mnist.npz'))
    xp = cp if device == 'gpu' and HAS_CUDA else np
    
    return (
        to_device(data['train_data'], device),
        to_device(data['train_labels'], device),
        to_device(data['test_data'], device),
        to_device(data['test_labels'], device)
    )

def compute_compositional_averages(data, labels, n_colors):
    """Compute class averages for each digit-color combination.
    
    Args:
        data: Array of shape (n_samples, 784 * (1 + n_colors))
        labels: Array of shape (n_samples, 2) with [digit, color] labels
        n_colors: Number of colors
        
    Returns:
        dict: Maps (digit, color) tuples to average input arrays
    """
    xp = cp if isinstance(data, cp.ndarray) else np
    averages = {}
    
    # Convert labels to same type as data if needed
    if isinstance(data, cp.ndarray) and isinstance(labels, np.ndarray):
        labels = cp.array(labels)
    elif isinstance(data, np.ndarray) and isinstance(labels, cp.ndarray):
        labels = np.array(labels)
    
    # Compute average for each digit-color combination
    for digit in range(10):
        for color in range(n_colors):
            # Find samples with this digit-color combination
            mask = (labels[:, 0] == digit) & (labels[:, 1] == color)
            if xp.any(mask):
                # Compute average of matching samples
                avg = xp.mean(data[mask], axis=0)
                averages[(digit, color)] = avg
    
    return averages

def compute_component_averages(data, labels, n_colors):
    """Compute separate averages for digits and colors.
    
    Args:
        data: Array of shape (n_samples, 784 * (1 + n_colors))
        labels: Array of shape (n_samples, 2) with [digit, color] labels
        n_colors: Number of colors
        
    Returns:
        tuple: (digit_averages, color_averages)
            - digit_averages: dict mapping digit to average
            - color_averages: dict mapping color to average
    """
    xp = cp if isinstance(data, cp.ndarray) else np
    digit_averages = {}
    color_averages = {}
    
    # Compute digit averages (averaging over all colors)
    for digit in range(10):
        mask = labels[:, 0] == digit
        if xp.any(mask):
            avg = xp.mean(data[mask], axis=0)
            digit_averages[digit] = avg
    
    # Compute color averages (averaging over all digits)
    for color in range(n_colors):
        mask = labels[:, 1] == color
        if xp.any(mask):
            avg = xp.mean(data[mask], axis=0)
            color_averages[color] = avg
    
    return digit_averages, color_averages

class SimpleDecoder:
    """Linear classifier for evaluation."""
    def __init__(self):
        self.W = None
        
    def fit(self, X, y):
        n_classes = int(y.max()) + 1  # Convert to int
        Y = np.zeros((len(y), n_classes))
        Y[np.arange(len(y)), y.astype(int)] = 1  # Ensure indices are integers
        self.W = np.linalg.pinv(X) @ Y
        
    def predict(self, X):
        return np.argmax(X @ self.W, axis=1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

def train_hebbian_colored_mnist(
    n_colors=2,  # Start with 2 colors (easy difficulty)
    hidden_size=300,  # Single hidden layer of size 300
    batch_size=100,
    eval_batch_size=1000,
    n_epochs=20,
    learning_rate=0.02,
    p=2.0,
    delta=0.4,
    k=2,
    allow_pathway_interaction=False,
    save_dir='results',
    train_ratio=0.8  # Add train_ratio parameter
):
    """Train network on colored MNIST with compositional signific inputs.
    
    Args:
        n_colors: Number of colors to use (2 for easy difficulty)
        hidden_size: Size of single hidden layer (300 as specified)
        batch_size: Training batch size
        eval_batch_size: Batch size for evaluation
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        p: Power for activation function
        delta: Anti-Hebbian learning strength
        k: Number of competing units
        allow_pathway_interaction: Whether to allow pathway interaction
        save_dir: Directory to save results
        train_ratio: Ratio of combinations to keep in training set
    """
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        config = locals()
        for key in sorted(config):
            if not key.startswith('_'):
                f.write(f'{key}: {config[key]}\n')
    
    # Load data
    print("Loading colored MNIST data...")
    train_data, train_labels, test_data, test_labels, holdout_pairs = generate_colored_mnist(
        n_colors=n_colors,
        device='gpu' if HAS_CUDA else 'cpu',
        include_signific=True,
        train_ratio=train_ratio
    )
    
    # Extract signific vectors
    train_signific = train_labels[:, 2:]  # Shape: (n_samples, 10 + n_colors)
    test_signific = test_labels[:, 2:]
    
    # Create network
    print("Initializing network...")
    layer_sizes = [784 * (1 + n_colors), hidden_size]  # Input -> Hidden
    signific_size = 10 + n_colors  # 10 digits + n_colors
    net = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],  # Connect to hidden layer
        p=p,
        delta=delta,
        k=k,
        device='gpu' if HAS_CUDA else 'cpu',
        allow_pathway_interaction=allow_pathway_interaction
    )
    
    # Training
    print("\nTraining network...")
    n_batches = len(train_data) // batch_size
    learning_rates = np.linspace(learning_rate, 0, n_epochs)  # Linear annealing
    print(f"Running for {n_epochs} epochs with {n_batches} batches per epoch")
    
    for epoch in trange(n_epochs, desc="Epochs"):
        # Shuffle data
        perm = cp.random.permutation(len(train_data)) if HAS_CUDA else np.random.permutation(len(train_data))
        train_data_epoch = train_data[perm]
        train_signific_epoch = train_signific[perm]
        
        # Batch training
        for batch in trange(n_batches, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_data = train_data_epoch[start_idx:end_idx]
            batch_signific = train_signific_epoch[start_idx:end_idx]
            net.update(batch_data, batch_signific, learning_rate=learning_rates[epoch])
        
        # Save weights visualization
        if (epoch + 1) % 5 == 0:
            fig = visualize_weights(net)
            plt.savefig(os.path.join(run_dir, f'weights_epoch_{epoch+1}.png'))
            plt.close(fig)
    
    print("\nEvaluating...")
    # Compute hidden representations for decoder
    train_repr = []
    for i in trange(0, len(train_data), eval_batch_size, desc="Computing train representations"):
        batch = train_data[i:i+eval_batch_size]
        h = to_device(net.forward(batch), 'cpu').T
        train_repr.append(h)
    train_repr = np.vstack(train_repr)
    
    test_repr = []
    for i in trange(0, len(test_data), eval_batch_size, desc="Computing test representations"):
        batch = test_data[i:i+eval_batch_size]
        h = to_device(net.forward(batch), 'cpu').T
        test_repr.append(h)
    test_repr = np.vstack(test_repr)
    
    # Move labels to CPU for evaluation
    train_labels_cpu = to_device(train_labels, 'cpu')
    test_labels_cpu = to_device(test_labels, 'cpu')
    
    # Train decoders for digits and colors
    print("Training decoders...")
    digit_decoder = SimpleDecoder()
    digit_decoder.fit(train_repr, train_labels_cpu[:, 0])  # First column is digit
    
    color_decoder = SimpleDecoder()
    color_decoder.fit(train_repr, train_labels_cpu[:, 1])  # Second column is color
    
    # Get decoder accuracies
    print("Computing accuracies...")
    train_digit_acc_decoder = digit_decoder.score(train_repr, train_labels_cpu[:, 0])
    test_digit_acc_decoder = digit_decoder.score(test_repr, test_labels_cpu[:, 0])
    train_color_acc_decoder = color_decoder.score(train_repr, train_labels_cpu[:, 1])
    test_color_acc_decoder = color_decoder.score(test_repr, test_labels_cpu[:, 1])
    
    # Compute full accuracy (both digit and color correct)
    train_full_acc_decoder = np.mean(
        (digit_decoder.predict(train_repr) == train_labels_cpu[:, 0]) &
        (color_decoder.predict(train_repr) == train_labels_cpu[:, 1])
    )
    test_full_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr) == test_labels_cpu[:, 0]) &
        (color_decoder.predict(test_repr) == test_labels_cpu[:, 1])
    )
    
    # Get RtS predictions
    train_preds = []
    for i in trange(0, len(train_data), eval_batch_size, desc="Computing train RtS predictions"):
        batch = train_data[i:i+eval_batch_size]
        digit_pred, color_pred = net.rts_classify(batch, [10, n_colors])
        train_preds.append([
            to_device(digit_pred, 'cpu'),
            to_device(color_pred, 'cpu')
        ])
    
    train_digit_preds = np.concatenate([p[0] for p in train_preds])
    train_color_preds = np.concatenate([p[1] for p in train_preds])
    
    test_preds = []
    for i in trange(0, len(test_data), eval_batch_size, desc="Computing test RtS predictions"):
        batch = test_data[i:i+eval_batch_size]
        digit_pred, color_pred = net.rts_classify(batch, [10, n_colors])
        test_preds.append([
            to_device(digit_pred, 'cpu'),
            to_device(color_pred, 'cpu')
        ])
    
    test_digit_preds = np.concatenate([p[0] for p in test_preds])
    test_color_preds = np.concatenate([p[1] for p in test_preds])
    
    # Identify OOD combinations by checking training data
    ood_indices = []
    for i, (digit, color) in enumerate([(d, c) for d in range(10) for c in range(n_colors)]):
        exists = False
        for d, c in zip(train_labels_cpu[:, 0], train_labels_cpu[:, 1]):
            if d == digit and c == color:
                exists = True
                break
        if not exists:
            ood_indices.append(i)
    
    # Compute RtS accuracies
    train_digit_acc_rts = np.mean(train_digit_preds == train_labels_cpu[:, 0])
    train_color_acc_rts = np.mean(train_color_preds == train_labels_cpu[:, 1])
    train_full_acc_rts = np.mean(
        (train_digit_preds == train_labels_cpu[:, 0]) & 
        (train_color_preds == train_labels_cpu[:, 1])
    )
    
    test_digit_acc_rts = np.mean(test_digit_preds == test_labels_cpu[:, 0])
    test_color_acc_rts = np.mean(test_color_preds == test_labels_cpu[:, 1])
    test_full_acc_rts = np.mean(
        (test_digit_preds == test_labels_cpu[:, 0]) & 
        (test_color_preds == test_labels_cpu[:, 1])
    )
    
    # Compute confusion matrices for both decoders and RtS
    train_digit_conf_decoder = compute_confusion_matrix(train_labels_cpu[:, 0], digit_decoder.predict(train_repr))
    train_color_conf_decoder = compute_confusion_matrix(train_labels_cpu[:, 1], color_decoder.predict(train_repr))
    test_digit_conf_decoder = compute_confusion_matrix(test_labels_cpu[:, 0], digit_decoder.predict(test_repr))
    test_color_conf_decoder = compute_confusion_matrix(test_labels_cpu[:, 1], color_decoder.predict(test_repr))
    
    train_digit_conf_rts = compute_confusion_matrix(train_labels_cpu[:, 0], train_digit_preds)
    train_color_conf_rts = compute_confusion_matrix(train_labels_cpu[:, 1], train_color_preds)
    test_digit_conf_rts = compute_confusion_matrix(test_labels_cpu[:, 0], test_digit_preds)
    test_color_conf_rts = compute_confusion_matrix(test_labels_cpu[:, 1], test_color_preds)
    
    # Compute compositional confusion matrices for RtS
    train_comp_conf_rts = compute_confusion_matrix(
        train_labels_cpu[:, 1] * 10 + train_labels_cpu[:, 0],  # color * 10 + digit for unique class ID
        train_color_preds * 10 + train_digit_preds
    )
    test_comp_conf_rts = compute_confusion_matrix(
        test_labels_cpu[:, 1] * 10 + test_labels_cpu[:, 0],
        test_color_preds * 10 + test_digit_preds
    )
    
    # Save confusion matrices
    plot_confusion_matrix(train_digit_conf_decoder, os.path.join(run_dir, 'train_confusion_digit_decoder.png'),
                         title='Training Confusion Matrix (Decoder, Digits)', normalize=True)
    plot_confusion_matrix(train_color_conf_decoder, os.path.join(run_dir, 'train_confusion_color_decoder.png'),
                         title='Training Confusion Matrix (Decoder, Colors)', normalize=True, is_color=True, n_colors=n_colors)
    plot_confusion_matrix(test_digit_conf_decoder, os.path.join(run_dir, 'test_confusion_digit_decoder.png'),
                         title='Test Confusion Matrix (Decoder, Digits)', normalize=True)
    plot_confusion_matrix(test_color_conf_decoder, os.path.join(run_dir, 'test_confusion_color_decoder.png'),
                         title='Test Confusion Matrix (Decoder, Colors)', normalize=True, is_color=True, n_colors=n_colors)
    
    plot_confusion_matrix(train_digit_conf_rts, os.path.join(run_dir, 'train_confusion_digit_rts.png'),
                         title='Training Confusion Matrix (RtS, Digits)', normalize=True)
    plot_confusion_matrix(train_color_conf_rts, os.path.join(run_dir, 'train_confusion_color_rts.png'),
                         title='Training Confusion Matrix (RtS, Colors)', normalize=True, is_color=True, n_colors=n_colors)
    plot_confusion_matrix(test_digit_conf_rts, os.path.join(run_dir, 'test_confusion_digit_rts.png'),
                         title='Test Confusion Matrix (RtS, Digits)', normalize=True)
    plot_confusion_matrix(test_color_conf_rts, os.path.join(run_dir, 'test_confusion_color_rts.png'),
                         title='Test Confusion Matrix (RtS, Colors)', normalize=True, is_color=True, n_colors=n_colors)
    
    # Plot compositional confusion matrices for RtS
    plot_confusion_matrix(train_comp_conf_rts, os.path.join(run_dir, 'train_confusion_compositional_rts.png'),
                         title='Training Confusion Matrix (RtS, Compositional)', normalize=True, 
                         is_compositional=True, n_colors=n_colors, ood_indices=ood_indices)
    plot_confusion_matrix(test_comp_conf_rts, os.path.join(run_dir, 'test_confusion_compositional_rts.png'),
                         title='Test Confusion Matrix (RtS, Compositional)', normalize=True,
                         is_compositional=True, n_colors=n_colors, ood_indices=ood_indices)
    
    # Compute class averages for each digit-color combination
    print("\nComputing compositional class averages...")
    class_averages = compute_compositional_averages(test_data, test_labels_cpu[:, :2], n_colors)
    
    # Create signific inputs for each digit-color combination
    signific_inputs = []
    for i, (digit, color) in enumerate([(d, c) for d in range(10) for c in range(n_colors)]):
        signific = np.zeros(10 + n_colors)
        signific[digit] = 1  # Set digit position
        signific[10 + color] = 1  # Set color position
        signific_inputs.append(signific)
    
    signific_inputs = to_device(np.stack(signific_inputs), net.device)
    
    # Compute StR-rep association matrix
    print("\nComputing StR-rep association matrix...")
    n_combinations = 10 * n_colors
    association_matrix = cp.zeros((n_combinations, n_combinations)) if HAS_CUDA else np.zeros((n_combinations, n_combinations))
    for i in range(n_combinations):
        for j in range(n_combinations):
            digit_i, color_i = i % 10, i // 10
            digit_j, color_j = j % 10, j // 10
            if (digit_j, color_j) in class_averages:  # Check if combination exists
                association_matrix[i, j] = net.str_rep(
                    signific_inputs[i:i+1],
                    class_averages[(digit_j, color_j)].reshape(1, -1)
                )
    
    # Plot association matrix
    fig = plt.figure(figsize=(10, 9))
    ax = plt.gca()
    # Convert to NumPy before plotting
    matrix_to_plot = to_device(association_matrix, 'cpu')
    plot_association_matrix(matrix_to_plot, os.path.join(run_dir, 'str_rep_association.png'),
                          title="StR-rep Association Matrix", n_colors=n_colors, ood_indices=ood_indices)
    
    # Compute StR-rec reconstructions
    print("\nComputing StR-rec reconstructions...")
    reconstructions = []
    reconstruction_similarities = []
    for i in trange(n_combinations, desc="Computing reconstructions"):
        digit, color = i % 10, i // 10
        if (digit, color) in class_averages:  # Only reconstruct if combination exists
            # Get reconstruction
            recon = net.str_rec(signific_inputs[i:i+1], max_iter=100)
            reconstructions.append(recon[0])
            
            # Compare with class average
            sim = cosine_similarity(
                [recon],
                [class_averages[(digit, color)].reshape(1, -1)],
                xp=cp if HAS_CUDA else np
            )
            reconstruction_similarities.append(float(to_device(sim, 'cpu')))
    
    # Stack reconstructions
    reconstructions = cp.stack(reconstructions) if HAS_CUDA else np.stack(reconstructions)
    
    # Plot reconstructions
    plot_reconstructions(
        reconstructions,
        np.stack([class_averages[(d, c)] for d, c in class_averages.keys()]),
        os.path.join(run_dir, 'str_rec_reconstructions.png'),
        n_colors=n_colors,
        ood_indices=ood_indices
    )
    
    # Plot reconstruction similarity matrix
    plot_reconstruction_similarity_matrix(
        reconstructions,
        np.stack([class_averages[(d, c)] for d, c in class_averages.keys()]),
        os.path.join(run_dir, 'str_rec_similarity.png'),
        n_colors=n_colors,
        ood_indices=ood_indices
    )
    
    # Print and save results
    print(f"\nResults:")
    print("Supervised Decoder:")
    print("Training:")
    print(f"Digit accuracy: {train_digit_acc_decoder:.4f}")
    print(f"Color accuracy: {train_color_acc_decoder:.4f}")
    print(f"Full accuracy: {train_full_acc_decoder:.4f}")
    print("\nTesting:")
    print(f"Digit accuracy: {test_digit_acc_decoder:.4f}")
    print(f"Color accuracy: {test_color_acc_decoder:.4f}")
    print(f"Full accuracy: {test_full_acc_decoder:.4f}")
    
    print("\nRtS Classification:")
    print("Training:")
    print(f"Digit accuracy: {train_digit_acc_rts:.4f}")
    print(f"Color accuracy: {train_color_acc_rts:.4f}")
    print(f"Full accuracy: {train_full_acc_rts:.4f}")
    print("\nTesting:")
    print(f"Digit accuracy: {test_digit_acc_rts:.4f}")
    print(f"Color accuracy: {test_color_acc_rts:.4f}")
    print(f"Full accuracy: {test_full_acc_rts:.4f}")
    
    # Save summary
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write("Colored MNIST Experiment Results\n")
        f.write("==============================\n\n")
        
        f.write("Supervised Decoder:\n")
        f.write("------------------\n")
        f.write("Training:\n")
        f.write(f"Digit accuracy: {train_digit_acc_decoder:.4f}\n")
        f.write(f"Color accuracy: {train_color_acc_decoder:.4f}\n")
        f.write(f"Full accuracy: {train_full_acc_decoder:.4f}\n\n")
        f.write("Testing:\n")
        f.write(f"Digit accuracy: {test_digit_acc_decoder:.4f}\n")
        f.write(f"Color accuracy: {test_color_acc_decoder:.4f}\n")
        f.write(f"Full accuracy: {test_full_acc_decoder:.4f}\n\n")
        
        f.write("RtS Classification:\n")
        f.write("------------------\n")
        f.write("Training:\n")
        f.write(f"Digit accuracy: {train_digit_acc_rts:.4f}\n")
        f.write(f"Color accuracy: {train_color_acc_rts:.4f}\n")
        f.write(f"Full accuracy: {train_full_acc_rts:.4f}\n\n")
        f.write("Testing:\n")
        f.write(f"Digit accuracy: {test_digit_acc_rts:.4f}\n")
        f.write(f"Color accuracy: {test_color_acc_rts:.4f}\n")
        f.write(f"Full accuracy: {test_full_acc_rts:.4f}\n\n")
        
        f.write("StR-rep Cosine Similarities:\n")
        f.write("---------------------------\n")
        
        # Calculate diagonal mean (matching digit-color pairs)
        diag_mean = float(to_device(
            cp.diag(association_matrix).mean() if HAS_CUDA 
            else np.diag(association_matrix).mean(), 'cpu'
        ))
        f.write(f"Average diagonal (matching pairs): {diag_mean:.4f}\n")
        
        # Calculate off-diagonal mean (non-matching pairs)
        off_diag_mean = float(to_device(
            (association_matrix.sum() - cp.diag(association_matrix).sum()) / 
            (n_combinations * n_combinations - n_combinations) if HAS_CUDA 
            else (association_matrix.sum() - np.diag(association_matrix).sum()) / 
            (n_combinations * n_combinations - n_combinations), 'cpu'
        ))
        f.write(f"Average off-diagonal (non-matching pairs): {off_diag_mean:.4f}\n")
        
        # Write full matrix
        f.write("\nFull association matrix:\n")
        matrix_cpu = to_device(association_matrix, 'cpu')
        for i in range(n_combinations):
            digit_i, color_i = i // n_colors, i % n_colors
            row_str = " ".join(f"{x:.4f}" for x in matrix_cpu[i])
            f.write(f"(d:{digit_i},c:{color_i}) {row_str}\n")
        
        f.write("\nStR-rec Reconstruction Similarities:\n")
        f.write("--------------------------------\n")
        f.write(f"Average similarity: {np.mean(reconstruction_similarities):.4f}\n")
        f.write("\nPer-combination similarities:\n")
        for i, sim in enumerate(reconstruction_similarities):
            digit, color = i // n_colors, i % n_colors
            f.write(f"Digit {digit}, Color {color}: {sim:.4f}\n")
    
    # Save results
    results = {
        'train_digit_acc_decoder': train_digit_acc_decoder,
        'train_color_acc_decoder': train_color_acc_decoder,
        'train_full_acc_decoder': train_full_acc_decoder,
        'test_digit_acc_decoder': test_digit_acc_decoder,
        'test_color_acc_decoder': test_color_acc_decoder,
        'test_full_acc_decoder': test_full_acc_decoder,
        'train_digit_acc_rts': train_digit_acc_rts,
        'train_color_acc_rts': train_color_acc_rts,
        'train_full_acc_rts': train_full_acc_rts,
        'test_digit_acc_rts': test_digit_acc_rts,
        'test_color_acc_rts': test_color_acc_rts,
        'test_full_acc_rts': test_full_acc_rts,
        'train_digit_conf_decoder': train_digit_conf_decoder,
        'train_color_conf_decoder': train_color_conf_decoder,
        'test_digit_conf_decoder': test_digit_conf_decoder,
        'test_color_conf_decoder': test_color_conf_decoder,
        'train_digit_conf_rts': train_digit_conf_rts,
        'train_color_conf_rts': train_color_conf_rts,
        'test_digit_conf_rts': test_digit_conf_rts,
        'test_color_conf_rts': test_color_conf_rts,
        'str_rep_association': to_device(association_matrix, 'cpu'),
        'str_rec_reconstructions': to_device(reconstructions, 'cpu'),
        'str_rec_similarities': reconstruction_similarities
    }
    np.save(os.path.join(run_dir, 'results.npy'), results)
    
    return net, results, run_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_colors', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--allow_pathway_interaction', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    net, results, run_dir = train_hebbian_colored_mnist(
        n_colors=args.n_colors,
        hidden_size=args.hidden_size,
        n_epochs=args.n_epochs,
        allow_pathway_interaction=args.allow_pathway_interaction,
        train_ratio=args.train_ratio
    ) 