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
    plot_reconstructions, plot_ood_reconstructions,
    plot_reconstruction_similarity_matrix, cosine_similarity, multichannel_to_rgb, COLORS, COLOR_LABELS, ColoredMNIST,
    CompositionTracker, compute_class_averages, plot_similarity_matrix,
    save_experiment_metrics, frobenius_norm_from_identity,
    set_random_seed
)
from core.config import NetworkConfig, TrainingConfig
import scipy.io
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from core.stats import test_accuracy, test_gmcc, test_frobenius, format_metric
from core.kh_decoder import Decoder
from core.simple_decoder import SimpleDecoder
from skimage.metrics import structural_similarity as ssim

set_random_seed()

# # Define RGB values for each color
# COLORS = {
#     3: np.array([  # RGB values for 3 colors
#         [1.0, 0.0, 0.0],  # Red
#         [0.0, 0.0, 1.0],  # Blue
#         [0.0, 1.0, 0.0],  # Green
#     ]),
#     5: np.array([  # RGB values for 5 colors
#         [1.0, 0.0, 0.0],  # Red
#         [0.0, 0.0, 1.0],  # Blue
#         [0.0, 0.8, 0.0],  # Green
#         [0.8, 0.0, 0.8],  # Purple
#         [1.0, 0.6, 0.0],  # Orange
#     ])
# }

# def multichannel_to_rgb(data, n_colors):
#     """Convert multi-channel representation to RGB image.
    
#     Args:
#         data: Array of shape (784 * (1 + n_colors),) or (batch_size, 784 * (1 + n_colors))
#         n_colors: Number of colors used in the dataset
        
#     Returns:
#         Array of shape (28, 28, 3) or (batch_size, 28, 28, 3) containing RGB values
#     """
#     if data.ndim == 1:
#         data = data.reshape(1, -1)
    
#     batch_size = data.shape[0]
#     xp = cp if isinstance(data, cp.ndarray) else np
#     colors = to_device(COLORS[n_colors], data.device if hasattr(data, 'device') else 'cpu')
    
#     # Extract intensity and color channels
#     intensity = data[:, :784].reshape(batch_size, 28, 28)
#     color_channels = [
#         data[:, (784 * (i + 1)):(784 * (i + 2))].reshape(batch_size, 28, 28)
#         for i in range(n_colors)
#     ]
    
#     # Initialize RGB output
#     rgb = xp.zeros((batch_size, 28, 28, 3))
    
#     # Combine channels using color mapping
#     for i, color_channel in enumerate(color_channels):
#         for j in range(3):  # RGB channels
#             rgb[:, :, :, j] += color_channel * colors[i, j]
    
#     # Scale by intensity
#     rgb = rgb * intensity.reshape(batch_size, 28, 28, 1)
    
#     # Ensure values are in [0, 1]
#     rgb = xp.clip(rgb, 0, 1)
    
#     return rgb[0] if data.shape[0] == 1 else rgb

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
    # Initialize ColoredMNIST handler
    mnist = ColoredMNIST(n_colors, device)
    
    # Load MNIST data
    mnist_data, mnist_labels, mnist_test_data, mnist_test_labels = load_mnist()
    xp = cp if device == 'gpu' and HAS_CUDA else np
    
    # Calculate number of combinations to hold out (round up to ensure we hold out enough)
    total_combinations = 10 * n_colors
    n_holdout = int(np.ceil(total_combinations * (1 - train_ratio)))
    
    # Generate holdout pairs if none specified
    if holdout_pairs is None:
        # Create list of available digits starting from 9
        available_digits = list(range(9, -1, -1))  # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        
        # Generate holdout pairs by alternating through colors
        holdout_pairs = []
        current_color = 0
        while len(holdout_pairs) < n_holdout and available_digits:
            digit = available_digits.pop(0)  # Take from the start (highest digits)
            holdout_pairs.append((digit, current_color))
            current_color = (current_color + 1) % n_colors  # Cycle through colors
        
        # Sort by digit then color for consistent order
        holdout_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)  # Sort descending by digit
    
    print(f"Holding out {len(holdout_pairs)} combinations: {', '.join([mnist.get_label(d, c) for d, c in holdout_pairs])}")
    
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

def format_gmcc(value, n_samples):
    """Format GMCC with significance stars."""
    if value is None:
        return "N/A"
    p_value = test_gmcc(value, n_samples)
    return format_metric(value, p_value)

def train_hebbian_colored_mnist(net_config, train_config):
    """Train network on colored MNIST with compositional signific inputs."""
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(train_config.save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        f.write("Network Configuration:\n")
        for key, value in vars(net_config).items():
            f.write(f"{key}: {value}\n")
        f.write("\nTraining Configuration:\n")
        for key, value in vars(train_config).items():
            f.write(f"{key}: {value}\n")
    
    # Load data
    print("Loading colored MNIST data...")
    train_data, train_labels, test_data, test_labels, holdout_pairs = generate_colored_mnist(
        n_colors=net_config.n_colors,
        device='gpu' if HAS_CUDA else 'cpu',
        include_signific=True,
        train_ratio=train_config.train_ratio
    )
    
    # Extract signific vectors
    train_signific = train_labels[:, 2:]  # Shape: (n_samples, 10 + n_colors)
    test_signific = test_labels[:, 2:]
    
    # Create network
    print("Initializing network...")
    net = HebbianNetwork(
        layer_sizes=net_config.layer_sizes,
        signific_size=net_config.signific_size,
        hidden_signific_connections=net_config.hidden_signific_connections,
        p=net_config.p,
        delta=net_config.delta,
        k=net_config.k,
        device='gpu' if HAS_CUDA else 'cpu',
        allow_pathway_interaction=net_config.allow_pathway_interaction,
        signific_p_multiplier=net_config.signific_p_multiplier
    )
    
    # Training
    print("\nTraining network...")
    n_batches = len(train_data) // train_config.batch_size
    learning_rates = np.linspace(train_config.learning_rate, 0, train_config.n_epochs)
    print(f"Running for {train_config.n_epochs} epochs with {n_batches} batches per epoch")
    
    for epoch in trange(train_config.n_epochs, desc="Epochs"):
        # Shuffle data
        perm = cp.random.permutation(len(train_data)) if HAS_CUDA else np.random.permutation(len(train_data))
        train_data_epoch = train_data[perm]
        train_signific_epoch = train_signific[perm]
        
        # Batch training
        for batch in trange(n_batches, desc=f"Epoch {epoch+1}/{train_config.n_epochs}", leave=False):
            start_idx = batch * train_config.batch_size
            end_idx = start_idx + train_config.batch_size
            batch_data = train_data_epoch[start_idx:end_idx]
            batch_signific = train_signific_epoch[start_idx:end_idx]
            net.update(batch_data, batch_signific, learning_rate=learning_rates[epoch])
        
        # # Save weights visualization
        # if (epoch + 1) % 5 == 0:
        #     fig = visualize_weights(net)
        #     plt.savefig(os.path.join(run_dir, f'weights_epoch_{epoch+1}.png'))
        #     plt.close(fig)
    
    print("\nEvaluating...")
    # Compute hidden representations for decoder
    train_repr = []
    for i in trange(0, len(train_data), train_config.eval_batch_size, desc="Computing train representations"):
        batch = train_data[i:i+train_config.eval_batch_size]
        # Get hidden layer activations instead of signific layer
        h = x = batch
        for layer in net.layers:
            h = layer.forward(h).T
        train_repr.append(to_device(h, 'cpu'))
    train_repr = np.vstack(train_repr)
    
    test_repr = []
    for i in trange(0, len(test_data), train_config.eval_batch_size, desc="Computing test representations"):
        batch = test_data[i:i+train_config.eval_batch_size]
        # Get hidden layer activations instead of signific layer
        h = x = batch
        for layer in net.layers:
            h = layer.forward(h).T
        test_repr.append(to_device(h, 'cpu'))
    test_repr = np.vstack(test_repr)
    
    # Move labels to CPU for evaluation
    train_labels_cpu = to_device(train_labels, 'cpu')
    test_labels_cpu = to_device(test_labels, 'cpu')
    
    # Train decoders for digits and colors
    print("Training decoders...")
    # digit_decoder = Decoder(input_dim=net_config.layer_sizes[-1], output_dim=10)
    digit_decoder = SimpleDecoder()
    digit_decoder.fit(train_repr, train_labels_cpu[:, 0].astype(np.int64))  # First column is digit
    
    # color_decoder = Decoder(input_dim=net_config.layer_sizes[-1], output_dim=net_config.n_colors)
    color_decoder = SimpleDecoder()
    color_decoder.fit(train_repr, train_labels_cpu[:, 1].astype(np.int64))  # Second column is color
    
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
    
    # Use larger batch size for evaluation
    eval_batch_size = train_config.eval_batch_size * 4  # Increase batch size for faster evaluation
    
    # Create ID/OOD indices first
    all_indices = set(range(net_config.n_colors * 10))
    holdout_indices = set(digit * net_config.n_colors + color for digit, color in holdout_pairs)
    train_indices = sorted(all_indices - holdout_indices)
    ood_indices = sorted(holdout_indices)
    id_indices = sorted(train_indices)

    # Create digit masks for ID/OOD
    digit_mask = np.zeros(10, dtype=bool)
    for idx in train_indices:
        digit = idx // net_config.n_colors
        digit_mask[digit] = True
    digit_ood_mask = ~digit_mask
    
    # Create color mask (all colors are always used)
    color_mask = np.array([True] * net_config.n_colors)
    
    # Create sample masks for test set
    test_id_mask = np.zeros(len(test_labels_cpu), dtype=bool)
    for i in range(len(test_labels_cpu)):
        comp_idx = test_labels_cpu[i, 0] * net_config.n_colors + test_labels_cpu[i, 1]
        if comp_idx in train_indices:
            test_id_mask[i] = True
    test_ood_mask = ~test_id_mask

      
    # Compute raw decoder accuracies (unconstrained)
    raw_full_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr) == test_labels_cpu[:, 0]) &
        (color_decoder.predict(test_repr) == test_labels_cpu[:, 1])
    )
    
    raw_id_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr[test_id_mask]) == test_labels_cpu[test_id_mask, 0]) &
        (color_decoder.predict(test_repr[test_id_mask]) == test_labels_cpu[test_id_mask, 1])
    )
    
    raw_ood_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr[test_ood_mask]) == test_labels_cpu[test_ood_mask, 0]) &
        (color_decoder.predict(test_repr[test_ood_mask]) == test_labels_cpu[test_ood_mask, 1])
    )
    
    raw_digit_acc_decoder = np.mean(digit_decoder.predict(test_repr) == test_labels_cpu[:, 0])
    raw_color_acc_decoder = np.mean(color_decoder.predict(test_repr) == test_labels_cpu[:, 1])
    
    raw_id_digit_acc_decoder = np.mean(digit_decoder.predict(test_repr[test_id_mask]) == test_labels_cpu[test_id_mask, 0])
    raw_id_color_acc_decoder = np.mean(color_decoder.predict(test_repr[test_id_mask]) == test_labels_cpu[test_id_mask, 1])
    
    raw_ood_digit_acc_decoder = np.mean(digit_decoder.predict(test_repr[test_ood_mask]) == test_labels_cpu[test_ood_mask, 0])
    raw_ood_color_acc_decoder = np.mean(color_decoder.predict(test_repr[test_ood_mask]) == test_labels_cpu[test_ood_mask, 1])
    
    
    # Get RtS predictions for training data
    train_digit_preds = []
    train_color_preds = []
    for i in trange(0, len(train_data), eval_batch_size, desc="Computing train RtS predictions"):
        batch = train_data[i:i+eval_batch_size]
        digit_pred, color_pred = net.rts_classify(batch, [10, net_config.n_colors])
        train_digit_preds.append(digit_pred)
        train_color_preds.append(color_pred)
    
    train_digit_preds = to_device(cp.concatenate(train_digit_preds) if HAS_CUDA else np.concatenate(train_digit_preds), 'cpu')
    train_color_preds = to_device(cp.concatenate(train_color_preds) if HAS_CUDA else np.concatenate(train_color_preds), 'cpu')
    
    # Get RtS predictions for test data
    test_digit_preds = []
    test_color_preds = []
    
    # Get unconstrained predictions first (using raw activations)
    unconstrained_test_preds = []
    for i in trange(0, len(test_data), eval_batch_size, desc="Computing unconstrained test predictions"):
        batch = test_data[i:i+eval_batch_size]
        # Get predictions using rts_classify
        digit_pred, color_pred = net.rts_classify(batch, [10, net_config.n_colors])
        
        # Convert to CPU if needed
        if HAS_CUDA:
            digit_pred = cp.asnumpy(digit_pred)
            color_pred = cp.asnumpy(color_pred)
        
        # Use predictions directly
        batch_digit_preds = digit_pred
        batch_color_preds = color_pred
        
        # Combine into compositional predictions
        batch_comp_preds = batch_digit_preds * net_config.n_colors + batch_color_preds
        unconstrained_test_preds.append(batch_comp_preds)
    
    unconstrained_test_preds = np.concatenate(unconstrained_test_preds)
    
    # Compute unconstrained confusion matrix
    unconstrained_test_comp_conf = compute_confusion_matrix(
        test_labels_cpu[:, 0] * net_config.n_colors + test_labels_cpu[:, 1],
        unconstrained_test_preds,
        num_classes=net_config.n_colors * 10
    )
    
    # Calculate true raw accuracies from unconstrained predictions
    true_test_comp_labels = test_labels_cpu[:, 0] * net_config.n_colors + test_labels_cpu[:, 1]
    
    # Overall raw accuracy (no masking or constraints)
    raw_full_acc = np.mean(unconstrained_test_preds == true_test_comp_labels)
    
    # Separate ID and OOD raw accuracies
    raw_id_acc = np.mean(unconstrained_test_preds[test_id_mask] == true_test_comp_labels[test_id_mask])
    raw_ood_acc = np.mean(unconstrained_test_preds[test_ood_mask] == true_test_comp_labels[test_ood_mask])
    
    # Get raw digit and color accuracies
    unconstrained_digit_preds = unconstrained_test_preds // net_config.n_colors
    unconstrained_color_preds = unconstrained_test_preds % net_config.n_colors
    
    raw_digit_acc = np.mean(unconstrained_digit_preds == test_labels_cpu[:, 0])
    raw_color_acc = np.mean(unconstrained_color_preds == test_labels_cpu[:, 1])
    
    # Separate ID/OOD digit and color accuracies
    raw_id_digit_acc = np.mean(unconstrained_digit_preds[test_id_mask] == test_labels_cpu[test_id_mask, 0])
    raw_id_color_acc = np.mean(unconstrained_color_preds[test_id_mask] == test_labels_cpu[test_id_mask, 1])
    raw_ood_digit_acc = np.mean(unconstrained_digit_preds[test_ood_mask] == test_labels_cpu[test_ood_mask, 0])
    raw_ood_color_acc = np.mean(unconstrained_color_preds[test_ood_mask] == test_labels_cpu[test_ood_mask, 1])
    
    # Split unconstrained confusion matrix for OOD
    unconstrained_test_comp_conf_ood = unconstrained_test_comp_conf[ood_indices][:, ood_indices]
    
    # Plot unconstrained OOD confusion matrix
    plot_confusion_matrix(unconstrained_test_comp_conf_ood,
                        os.path.join(run_dir, 'unconstrained_confusion_compositional_ood.png'),
                        normalize=True, is_compositional=True,
                        n_colors=net_config.n_colors, train_indices=train_indices,
                        title_prefix="Colored MNIST: ")
    
    # Get all possible OOD digits and colors once
    ood_digits = set(idx // net_config.n_colors for idx in ood_indices)
    ood_colors = set(idx % net_config.n_colors for idx in ood_indices)
    
    # Get all possible ID digits and colors
    id_digits = set(idx // net_config.n_colors for idx in train_indices)
    id_colors = set(idx % net_config.n_colors for idx in train_indices)
    
    for i in trange(0, len(test_data), eval_batch_size, desc="Computing test RtS predictions"):
        batch = test_data[i:i+eval_batch_size]
        digit_pred, color_pred = net.rts_classify(batch, [10, net_config.n_colors])
        
        # For this batch, identify which samples are OOD
        batch_ood_mask = np.zeros(len(batch), dtype=bool)
        for j in range(len(batch)):
            sample_idx = i + j
            comp_idx = test_labels_cpu[sample_idx, 0] * net_config.n_colors + test_labels_cpu[sample_idx, 1]
            if comp_idx in ood_indices:
                batch_ood_mask[j] = True
        
        # Convert predictions to CPU if needed
        if HAS_CUDA:
            digit_pred = cp.asnumpy(digit_pred)
            color_pred = cp.asnumpy(color_pred)
        
        # For each sample, get predictions based on ID/OOD status
        batch_digit_preds = []
        batch_color_preds = []
        
        for j in range(len(batch)):
            if batch_ood_mask[j]:
                # For OOD samples, only consider OOD digits/colors
                if digit_pred[j] not in ood_digits or color_pred[j] not in ood_colors:
                    # If prediction is not in OOD set, pick closest valid OOD prediction
                    valid_digits = list(ood_digits)
                    valid_colors = list(ood_colors)
                    digit_pred[j] = valid_digits[0]  # Default to first valid digit
                    color_pred[j] = valid_colors[0]  # Default to first valid color
            else:
                # For ID samples, only consider ID digits/colors
                if digit_pred[j] not in id_digits or color_pred[j] not in id_colors:
                    # If prediction is not in ID set, pick closest valid ID prediction
                    valid_digits = list(id_digits)
                    valid_colors = list(id_colors)
                    digit_pred[j] = valid_digits[0]  # Default to first valid digit
                    color_pred[j] = valid_colors[0]  # Default to first valid color
            
            batch_digit_preds.append(digit_pred[j])
            batch_color_preds.append(color_pred[j])
        
        test_digit_preds.append(np.array(batch_digit_preds))
        test_color_preds.append(np.array(batch_color_preds))
    
    test_digit_preds = np.concatenate(test_digit_preds)
    test_color_preds = np.concatenate(test_color_preds)
    
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
    
    # Compute accuracies for ID/OOD splits
    id_acc_rts = np.mean(
        (test_digit_preds[test_id_mask] == test_labels_cpu[test_id_mask, 0]) &
        (test_color_preds[test_id_mask] == test_labels_cpu[test_id_mask, 1])
    )
    
    ood_acc_rts = np.mean(
        (test_digit_preds[test_ood_mask] == test_labels_cpu[test_ood_mask, 0]) &
        (test_color_preds[test_ood_mask] == test_labels_cpu[test_ood_mask, 1])
    )
    
    id_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr)[test_id_mask] == test_labels_cpu[test_id_mask, 0]) &
        (color_decoder.predict(test_repr)[test_id_mask] == test_labels_cpu[test_id_mask, 1])
    )
    
    ood_acc_decoder = np.mean(
        (digit_decoder.predict(test_repr)[test_ood_mask] == test_labels_cpu[test_ood_mask, 0]) &
        (color_decoder.predict(test_repr)[test_ood_mask] == test_labels_cpu[test_ood_mask, 1])
    )
    
    # Compute basic confusion matrices for digits and colors
    train_digit_conf_rts = compute_confusion_matrix(train_labels_cpu[:, 0], train_digit_preds)
    train_color_conf_rts = compute_confusion_matrix(train_labels_cpu[:, 1], train_color_preds)
    test_digit_conf_rts = compute_confusion_matrix(test_labels_cpu[:, 0], test_digit_preds)
    test_color_conf_rts = compute_confusion_matrix(test_labels_cpu[:, 1], test_color_preds)
    
    # Compute decoder confusion matrices
    train_digit_conf_decoder = compute_confusion_matrix(train_labels_cpu[:, 0], digit_decoder.predict(train_repr))
    train_color_conf_decoder = compute_confusion_matrix(train_labels_cpu[:, 1], color_decoder.predict(train_repr))
    test_digit_conf_decoder = compute_confusion_matrix(test_labels_cpu[:, 0], digit_decoder.predict(test_repr))
    test_color_conf_decoder = compute_confusion_matrix(test_labels_cpu[:, 1], color_decoder.predict(test_repr))
    
    # Split digit confusion matrices
    test_digit_conf_id = test_digit_conf_rts[digit_mask][:, digit_mask]
    test_digit_conf_ood = test_digit_conf_rts[digit_ood_mask][:, digit_ood_mask]
    test_digit_conf_decoder_id = test_digit_conf_decoder[digit_mask][:, digit_mask]
    test_digit_conf_decoder_ood = test_digit_conf_decoder[digit_ood_mask][:, digit_ood_mask]
    
    # Split color confusion matrices (all colors are used)
    test_color_conf_id = test_color_conf_rts
    test_color_conf_ood = np.zeros((0, 0))  # No OOD colors
    test_color_conf_decoder_id = test_color_conf_decoder
    test_color_conf_decoder_ood = np.zeros((0, 0))  # No OOD colors
    
    # Compute compositional predictions
    train_comp_preds_rts = train_digit_preds * net_config.n_colors + train_color_preds
    test_comp_preds_rts = test_digit_preds * net_config.n_colors + test_color_preds
    train_comp_preds_decoder = digit_decoder.predict(train_repr) * net_config.n_colors + color_decoder.predict(train_repr)
    test_comp_preds_decoder = digit_decoder.predict(test_repr) * net_config.n_colors + color_decoder.predict(test_repr)
    
    # Compute compositional confusion matrices for all predictions (ID & OOD)
    test_comp_conf_rts = compute_confusion_matrix(
        test_labels_cpu[:, 0] * net_config.n_colors + test_labels_cpu[:, 1],
        test_comp_preds_rts,
        num_classes=net_config.n_colors * 10
    )
    test_comp_conf_decoder = compute_confusion_matrix(
        test_labels_cpu[:, 0] * net_config.n_colors + test_labels_cpu[:, 1],
        test_comp_preds_decoder,
        num_classes=net_config.n_colors * 10
    )
    
    # Split into ID and OOD
    test_comp_conf_rts_id = test_comp_conf_rts[id_indices][:, id_indices]
    test_comp_conf_rts_ood = test_comp_conf_rts[ood_indices][:, ood_indices]
    test_comp_conf_decoder_id = test_comp_conf_decoder[id_indices][:, id_indices]
    test_comp_conf_decoder_ood = test_comp_conf_decoder[ood_indices][:, ood_indices]
    
    # Compute constrained predictions for both ID and OOD
    constrained_test_preds_rts_id = []
    constrained_test_preds_rts_ood = []
    constrained_test_preds_decoder_id = []
    constrained_test_preds_decoder_ood = []
    
    # Get constrained RtS predictions
    for i in trange(0, len(test_data), eval_batch_size, desc="Computing constrained RtS predictions"):
        batch = test_data[i:i+eval_batch_size]
        # Get predictions using rts_classify
        digit_pred, color_pred = net.rts_classify(batch, [10, net_config.n_colors])
        
        # Convert to CPU if needed
        if HAS_CUDA:
            digit_pred = cp.asnumpy(digit_pred)
            color_pred = cp.asnumpy(color_pred)
        
        # For each sample, get predictions based on ID/OOD status
        for j in range(len(batch)):
            sample_idx = i + j
            comp_idx = test_labels_cpu[sample_idx, 0] * net_config.n_colors + test_labels_cpu[sample_idx, 1]
            
            if comp_idx in id_indices:
                # For ID samples, only consider ID digits/colors
                if digit_pred[j] not in id_digits or color_pred[j] not in id_colors:
                    # If prediction is not in ID set, pick closest valid ID prediction
                    valid_digits = list(id_digits)
                    valid_colors = list(id_colors)
                    digit_pred[j] = valid_digits[0]  # Default to first valid digit
                    color_pred[j] = valid_colors[0]  # Default to first valid color
                constrained_test_preds_rts_id.append(digit_pred[j] * net_config.n_colors + color_pred[j])
            else:
                # For OOD samples, only consider OOD digits/colors
                if digit_pred[j] not in ood_digits or color_pred[j] not in ood_colors:
                    # If prediction is not in OOD set, pick closest valid OOD prediction
                    valid_digits = list(ood_digits)
                    valid_colors = list(ood_colors)
                    digit_pred[j] = valid_digits[0]  # Default to first valid digit
                    color_pred[j] = valid_colors[0]  # Default to first valid color
                constrained_test_preds_rts_ood.append(digit_pred[j] * net_config.n_colors + color_pred[j])
    
    # # Get constrained decoder predictions
    # for i in trange(0, len(test_data), eval_batch_size, desc="Computing constrained decoder predictions"):
    #     batch = test_data[i:i+eval_batch_size]
    #     batch_repr = net.forward(batch).T
        
    #     # Convert to CPU for decoder predictions
    #     if HAS_CUDA:
    #         batch_repr = cp.asnumpy(batch_repr)
        
    #     # Get raw predictions
    #     digit_scores = batch_repr @ digit_decoder.W
    #     color_scores = batch_repr @ color_decoder.W
        
    #     for j in range(len(batch)):
    #         sample_idx = i + j
    #         comp_idx = test_labels_cpu[sample_idx, 0] * net_config.n_colors + test_labels_cpu[sample_idx, 1]
            
    #         if comp_idx in id_indices:
    #             # For ID samples, only consider ID digits/colors
    #             digit_mask = np.zeros_like(digit_scores[j], dtype=bool)
    #             for d in id_digits:
    #                 digit_mask[d] = True
    #             digit_scores[j][~digit_mask] = -np.inf
                
    #             color_mask = np.zeros_like(color_scores[j], dtype=bool)
    #             for c in id_colors:
    #                 color_mask[c] = True
    #             color_scores[j][~color_mask] = -np.inf
                
    #             # Get predictions from masked scores
    #             digit_pred = np.argmax(digit_scores[j])
    #             color_pred = np.argmax(color_scores[j])
    #             constrained_test_preds_decoder_id.append(digit_pred * net_config.n_colors + color_pred)
    #         else:
    #             # For OOD samples, only consider OOD digits/colors
    #             digit_mask = np.zeros_like(digit_scores[j], dtype=bool)
    #             for d in ood_digits:
    #                 digit_mask[d] = True
    #             digit_scores[j][~digit_mask] = -np.inf
                
    #             color_mask = np.zeros_like(color_scores[j], dtype=bool)
    #             for c in ood_colors:
    #                 color_mask[c] = True
    #             color_scores[j][~color_mask] = -np.inf
                
    #             # Get predictions from masked scores
    #             digit_pred = np.argmax(digit_scores[j])
    #             color_pred = np.argmax(color_scores[j])
    #             constrained_test_preds_decoder_ood.append(digit_pred * net_config.n_colors + color_pred)
    
    # # Compute constrained confusion matrices for RtS
    # constrained_test_comp_conf_rts_id = compute_confusion_matrix(
    #     test_labels_cpu[test_id_mask][:, 0] * net_config.n_colors + test_labels_cpu[test_id_mask][:, 1],
    #     np.array(constrained_test_preds_rts_id),
    #     num_classes=net_config.n_colors * 10
    # )[id_indices][:, id_indices]
    
    # constrained_test_comp_conf_rts_ood = compute_confusion_matrix(
    #     test_labels_cpu[test_ood_mask][:, 0] * net_config.n_colors + test_labels_cpu[test_ood_mask][:, 1],
    #     np.array(constrained_test_preds_rts_ood),
    #     num_classes=net_config.n_colors * 10
    # )[ood_indices][:, ood_indices]
    
    # # Compute constrained confusion matrices for decoder
    # constrained_test_comp_conf_decoder_id = compute_confusion_matrix(
    #     test_labels_cpu[test_id_mask][:, 0] * net_config.n_colors + test_labels_cpu[test_id_mask][:, 1],
    #     np.array(constrained_test_preds_decoder_id),
    #     num_classes=net_config.n_colors * 10
    # )[id_indices][:, id_indices]
    
    # constrained_test_comp_conf_decoder_ood = compute_confusion_matrix(
    #     test_labels_cpu[test_ood_mask][:, 0] * net_config.n_colors + test_labels_cpu[test_ood_mask][:, 1],
    #     np.array(constrained_test_preds_decoder_ood),
    #     num_classes=net_config.n_colors * 10
    # )[ood_indices][:, ood_indices]

    # Before selecting OOD indices, normalize the full matrix
    row_sums = unconstrained_test_comp_conf.sum(axis=1, keepdims=True)
    normalized_full_conf = unconstrained_test_comp_conf / row_sums
    # Then select OOD portion
    normalized_ood_conf = normalized_full_conf[ood_indices][:, ood_indices]
    
    # Save confusion matrices
    # Plot WITHOUT additional normalization
    plot_confusion_matrix(normalized_ood_conf,
                        os.path.join(run_dir, 'full_normalized_confusion_compositional_ood.png'),
                        normalize=False,  # Important: don't normalize again
                        is_compositional=True,
                        n_colors=net_config.n_colors,
                        train_indices=train_indices,
                        title_prefix="Colored MNIST: ")
    plot_confusion_matrix(train_digit_conf_rts, os.path.join(run_dir, 'rts_confusion_digit_train.png'),
                         normalize=True, title_prefix="Colored MNIST: ")
    plot_confusion_matrix(train_color_conf_rts, os.path.join(run_dir, 'rts_confusion_color_train.png'),
                         normalize=True, is_color=True, n_colors=net_config.n_colors, title_prefix="Colored MNIST: ")
    plot_confusion_matrix(test_digit_conf_rts, os.path.join(run_dir, 'rts_confusion_digit_test.png'),
                         normalize=True, title_prefix="Colored MNIST: ")
    plot_confusion_matrix(test_color_conf_rts, os.path.join(run_dir, 'rts_confusion_color_test.png'),
                         normalize=True, is_color=True, n_colors=net_config.n_colors, title_prefix="Colored MNIST: ")
    
    # Plot full compositional confusion matrices (all compositions)
    plot_confusion_matrix(test_comp_conf_rts, os.path.join(run_dir, 'rts_confusion_compositional_all.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors, 
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    plot_confusion_matrix(test_comp_conf_decoder, os.path.join(run_dir, 'decoder_confusion_compositional_all.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors,
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    # Plot ID confusion matrices
    plot_confusion_matrix(test_comp_conf_rts_id, os.path.join(run_dir, 'rts_confusion_compositional_id.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors,
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    plot_confusion_matrix(test_comp_conf_decoder_id, os.path.join(run_dir, 'decoder_confusion_compositional_id.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors,
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    # Plot OOD confusion matrices
    plot_confusion_matrix(test_comp_conf_rts_ood, os.path.join(run_dir, 'rts_confusion_compositional_ood.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors,
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    plot_confusion_matrix(test_comp_conf_decoder_ood, os.path.join(run_dir, 'decoder_confusion_compositional_ood.png'),
                         normalize=True, is_compositional=True, n_colors=net_config.n_colors,
                         train_indices=train_indices, title_prefix="Colored MNIST: ")
    
    # After computing compositional class averages...
    print("\nComputing compositional class averages...")
    class_averages = compute_compositional_averages(test_data, test_labels_cpu[:, :2], net_config.n_colors)

    # Create signific inputs for each digit-color combination
    signific_inputs = []
    for digit in range(10):
        for color in range(net_config.n_colors):
            signific = np.zeros(10 + net_config.n_colors)
            signific[digit] = 1  # Set digit position
            signific[10 + color] = 1  # Set color position
            signific_inputs.append(signific)

    signific_inputs = to_device(np.stack(signific_inputs), net.device)

    # Compute StR-rep association matrix
    print("\nComputing StR-rep association matrix...")
    n_combinations = 10 * net_config.n_colors
    similarity_matrix = cp.zeros((n_combinations, n_combinations)) if HAS_CUDA else np.zeros((n_combinations, n_combinations))
    for i in range(n_combinations):
        digit_i, color_i = i // net_config.n_colors, i % net_config.n_colors
        for j in range(n_combinations):
            digit_j, color_j = j // net_config.n_colors, j % net_config.n_colors
            if (digit_j, color_j) in class_averages:  # Check if combination exists
                similarity_matrix[i, j] = net.str_rep(
                    signific_inputs[i:i+1],
                    class_averages[(digit_j, color_j)].reshape(1, -1)
                )

    # Compute StR-rec reconstructions
    print("\nComputing StR-rec reconstructions...")
    reconstructions = []
    reconstruction_similarities = []
    for i in trange(n_combinations, desc="Computing reconstructions"):
        digit, color = i // net_config.n_colors, i % net_config.n_colors
        if (digit, color) in class_averages:  # Only reconstruct if combination exists
            # Get reconstruction
            recon = net.str_rec(signific_inputs[i:i+1])
            reconstructions.append(recon[0])
            
            # Compare with class average using SSIM
            recon_img = to_device(recon[0], 'cpu')
            if isinstance(recon_img, cp.ndarray):
                recon_img = cp.asnumpy(recon_img)
            recon_img = recon_img.reshape(28, 28, -1)
            
            avg_img = to_device(class_averages[(digit, color)], 'cpu')
            if isinstance(avg_img, cp.ndarray):
                avg_img = cp.asnumpy(avg_img)
            avg_img = avg_img.reshape(28, 28, -1)
            
            sim = ssim(recon_img, avg_img, data_range=1.0, channel_axis=2)  # Use channel_axis for colored images
            reconstruction_similarities.append(float(sim))

    # Stack reconstructions
    reconstructions = cp.stack(reconstructions) if HAS_CUDA else np.stack(reconstructions)

    # Convert class_averages to list in same order as reconstructions
    class_averages_list = []
    for i in range(n_combinations):
        digit, color = i // net_config.n_colors, i % net_config.n_colors
        if (digit, color) in class_averages:
            class_averages_list.append(class_averages[(digit, color)])
    class_averages_list = cp.stack(class_averages_list) if HAS_CUDA else np.stack(class_averages_list)

    # Plot reconstructions
    plot_reconstructions(reconstructions, class_averages_list, os.path.join(run_dir, 'str_reconstructions.png'),
                        n_colors=net_config.n_colors, train_indices=train_indices,
                        title_prefix="Colored MNIST: ")
    
    # Plot OOD reconstructions
    plot_ood_reconstructions(reconstructions, class_averages_list, os.path.join(run_dir, 'str_ood_reconstructions.png'),
                           n_colors=net_config.n_colors, train_indices=train_indices,
                           title_prefix="Colored MNIST: ")

    # Plot reconstruction similarity matrices
    plot_reconstruction_similarity_matrix(reconstructions, class_averages_list, os.path.join(run_dir, 'str-rec_association_compositional_test.png'),
                                            n_colors=net_config.n_colors, train_indices=train_indices,
                                            title_prefix="Colored MNIST: ")
    
    # Compute Str-rec Similarity Metrics
    str_rec_matrix = np.zeros((10 * net_config.n_colors, 10 * net_config.n_colors))
    for i in range(10 * net_config.n_colors):
        recon_i = to_device(reconstructions[i], 'cpu')
        if isinstance(recon_i, cp.ndarray):
            recon_i = cp.asnumpy(recon_i)
        recon_i = recon_i.reshape(28, 28, -1)
        
        for j in range(10 * net_config.n_colors):
            avg_j = to_device(class_averages_list[j], 'cpu')
            if isinstance(avg_j, cp.ndarray):
                avg_j = cp.asnumpy(avg_j)
            avg_j = avg_j.reshape(28, 28, -1)
            str_rec_matrix[i, j] = ssim(recon_i, avg_j, data_range=1.0, channel_axis=2)
    
    # Compute Str-rec Similarity Metrics
    str_rec_diag_sim = np.diag(str_rec_matrix).mean()
    str_rec_off_diag_sim = str_rec_matrix[~np.eye(str_rec_matrix.shape[0], dtype=bool)].mean()
    
    # Create ID and OOD masks
    # train_indices = train_config.train_indices  # Ensure this is correctly sourced
    id_mask = np.zeros(str_rec_matrix.shape[0], dtype=bool)
    id_mask[train_indices] = True
    ood_mask = ~id_mask
    
    # Compute similarities for ID
    str_rec_diag_sim_id = np.diag(str_rec_matrix[id_mask][:, id_mask]).mean()
    str_rec_off_diag_sim_id = str_rec_matrix[id_mask][:, id_mask][~np.eye(np.sum(id_mask), dtype=bool)].mean()
    
    # Compute similarities for OOD
    str_rec_diag_sim_ood = np.diag(str_rec_matrix[ood_mask][:, ood_mask]).mean()
    str_rec_off_diag_sim_ood = str_rec_matrix[ood_mask][:, ood_mask][~np.eye(np.sum(ood_mask), dtype=bool)].mean()
    
    # Save summary with all metrics
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write("Colored MNIST Experiment Results\n")
        f.write("==============================\n\n")
        
        # StR-rep metrics
        f.write("StR-rep Metrics:\n")
        f.write("---------------\n")
        str_rep_fn = frobenius_norm_from_identity(similarity_matrix, return_numpy=True)
        str_rep_fn_p = test_frobenius(str_rep_fn, 10 * net_config.n_colors)  # Test against full matrix size
        
        # Create normalized version
        norm_str_rep = similarity_matrix.copy()
        for i in range(norm_str_rep.shape[0]):
            row_min = norm_str_rep[i].min()
            row_max = norm_str_rep[i].max()
            if row_max > row_min:
                norm_str_rep[i] = (norm_str_rep[i] - row_min) / (row_max - row_min)
        
        str_rep_fn_norm = frobenius_norm_from_identity(norm_str_rep, return_numpy=True)
        str_rep_fn_norm_p = test_frobenius(str_rep_fn_norm, 10 * net_config.n_colors)
        
        f.write(f"StR-rep FN: {format_metric(str_rep_fn, str_rep_fn_p)}\n")
        f.write(f"StR-rep FN Normalized: {format_metric(str_rep_fn_norm, str_rep_fn_norm_p)}\n\n")
        
        # Str-rec Metrics
        f.write("Str-rec Similarity Metrics:\n")
        f.write("---------------------------\n")
        f.write(f"Str-rec Diagonal Similarity: {str_rec_diag_sim:.4f}\n")
        f.write(f"Str-rec Off-diagonal Similarity: {str_rec_off_diag_sim:.4f}\n")
        f.write(f"Str-rec Diagonal Similarity (ID): {str_rec_diag_sim_id:.4f}\n")
        f.write(f"Str-rec Off-diagonal Similarity (ID): {str_rec_off_diag_sim_id:.4f}\n")
        f.write(f"Str-rec Diagonal Similarity (OOD): {str_rec_diag_sim_ood:.4f}\n")
        f.write(f"Str-rec Off-diagonal Similarity (OOD): {str_rec_off_diag_sim_ood:.4f}\n\n")
        
        # Classification metrics
        f.write("Constrained Classification Metrics:\n")
        f.write("---------------------\n")
        
        # RtS Classification
        f.write("\nRtS Classification:\n")
        
        # Digit accuracies
        train_digit_acc_p = test_accuracy(train_digit_acc_rts, 10, len(train_data))
        test_digit_acc_p = test_accuracy(test_digit_acc_rts, 10, len(test_data))
        f.write(f"ID Digit Accuracy: {format_metric(train_digit_acc_rts, train_digit_acc_p)}\n")
        f.write(f"OOD Digit Accuracy: {format_metric(test_digit_acc_rts, test_digit_acc_p)}\n")
        
        # Color accuracies
        train_color_acc_p = test_accuracy(train_color_acc_rts, net_config.n_colors, len(train_data))
        test_color_acc_p = test_accuracy(test_color_acc_rts, net_config.n_colors, len(test_data))
        f.write(f"ID Color Accuracy: {format_metric(train_color_acc_rts, train_color_acc_p)}\n")
        f.write(f"OOD Color Accuracy: {format_metric(test_color_acc_rts, test_color_acc_p)}\n")
        
        # Full compositional accuracies
        train_full_acc_p = test_accuracy(train_full_acc_rts, 10 * net_config.n_colors, len(train_data))
        test_full_acc_p = test_accuracy(test_full_acc_rts, 10 * net_config.n_colors, len(test_data))
        f.write(f"ID Full Accuracy: {format_metric(train_full_acc_rts, train_full_acc_p)}\n")
        f.write(f"OOD Full Accuracy: {format_metric(test_full_acc_rts, test_full_acc_p)}\n")

                
        f.write("\nRaw Performance Metrics (No Masking/Constraints):\n")
        f.write("----------------------------------------\n")
        f.write(f"RtS Overall Raw Accuracy: {raw_full_acc:.4f}\n")
        f.write(f"RtS Raw ID Accuracy: {raw_id_acc:.4f}\n")
        f.write(f"RtS Raw OOD Accuracy: {raw_ood_acc:.4f}\n")
        f.write(f"RtS Raw Digit Accuracy: {raw_digit_acc:.4f}\n")
        f.write(f"RtS Raw Color Accuracy: {raw_color_acc:.4f}\n")
        f.write(f"RtS Raw ID Digit Accuracy: {raw_id_digit_acc:.4f}\n")
        f.write(f"RtS Raw ID Color Accuracy: {raw_id_color_acc:.4f}\n")
        f.write(f"RtS Raw OOD Digit Accuracy: {raw_ood_digit_acc:.4f}\n")
        f.write(f"RtS Raw OOD Color Accuracy: {raw_ood_color_acc:.4f}\n\n")
        f.write(f"Decoder Overall Raw Accuracy: {raw_full_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw ID Accuracy: {raw_id_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw OOD Accuracy: {raw_ood_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw Digit Accuracy: {raw_digit_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw Color Accuracy: {raw_color_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw ID Digit Accuracy: {raw_id_digit_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw ID Color Accuracy: {raw_id_color_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw OOD Digit Accuracy: {raw_ood_digit_acc_decoder:.4f}\n")
        f.write(f"Decoder Raw OOD Color Accuracy: {raw_ood_color_acc_decoder:.4f}\n\n")
        
        # Detailed matrices
        f.write("\nDetailed Matrices:\n")
        f.write("=================\n\n")
        
        f.write("StR-rep Matrix:\n")
        for i in range(similarity_matrix.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in similarity_matrix[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("StR-rec Matrix:\n")
        for i in range(str_rec_matrix.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in str_rec_matrix[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("RtS Confusion Matrix (Test):\n")
        for i in range(test_comp_conf_rts.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in test_comp_conf_rts[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("Decoder Confusion Matrix (Test):\n")
        for i in range(test_comp_conf_decoder.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in test_comp_conf_decoder[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("Unconstrained Confusion Matrix (Test):\n")
        for i in range(unconstrained_test_comp_conf.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in unconstrained_test_comp_conf[i])
            f.write(f"{row_str}\n")
        f.write("\n")
    
    # Assemble results after all computations
    results = {
        'str_rec_matrix': to_device(str_rec_matrix, 'cpu'),
        'str_rec_diag_sim': str_rec_diag_sim,
        'str_rec_off_diag_sim': str_rec_off_diag_sim,
        'str_rec_diag_sim_id': str_rec_diag_sim_id,
        'str_rec_off_diag_sim_id': str_rec_off_diag_sim_id,
        'str_rec_diag_sim_ood': str_rec_diag_sim_ood,
        'str_rec_off_diag_sim_ood': str_rec_off_diag_sim_ood,
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
        'train_comp_conf_rts': test_comp_conf_rts,
        'test_comp_conf_rts': test_comp_conf_rts,
        'train_comp_conf_decoder': test_comp_conf_decoder,
        'test_comp_conf_decoder': test_comp_conf_decoder,
        'str_rep_similarity': to_device(similarity_matrix, 'cpu'),
        'str_rec_reconstructions': to_device(reconstructions, 'cpu'),
        'str_rec_similarities': reconstruction_similarities,
        'str_rec_matrix': to_device(str_rec_matrix, 'cpu'),
        'train_indices': train_indices,
        'unconstrained_test_comp_conf': unconstrained_test_comp_conf,
        'unconstrained_test_comp_conf_ood': normalized_ood_conf,
        'raw_full_acc': raw_full_acc,
        'raw_id_acc': raw_id_acc,
        'raw_ood_acc': raw_ood_acc,
        'raw_digit_acc': raw_digit_acc,
        'raw_color_acc': raw_color_acc,
        'raw_id_digit_acc': raw_id_digit_acc,
        'raw_id_color_acc': raw_id_color_acc,
        'raw_ood_digit_acc': raw_ood_digit_acc,
        'raw_ood_color_acc': raw_ood_color_acc
    }
    np.save(os.path.join(run_dir, 'results.npy'), results)
    
    # Save detailed metrics
    train_config.train_indices = train_indices  # Add train indices to config for metrics
    save_experiment_metrics(run_dir, net_config, train_config, results)
    
    return net, results, run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_size', type=int, default=200)  # Increased for more capacity
    parser.add_argument('--p', type=float, default=3.0)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--delta', type=float, default=0.4)
    parser.add_argument('--signific_p_multiplier', type=float, default=4.0)  # Matched with MNIST
    parser.add_argument('--allow_pathway_interaction', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--n_epochs', type=int, default=50)  # Matched with MNIST
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    parser.add_argument('--n_colors', type=int, default=2)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    args = parser.parse_args()
    
    set_random_seed(args.seed)
    
    net_config = NetworkConfig(
        hidden_sizes=args.hidden_size,
        n_colors=args.n_colors,
        p=args.p,
        k=args.k,
        delta=args.delta,
        signific_p_multiplier=args.signific_p_multiplier,
        allow_pathway_interaction=args.allow_pathway_interaction
    )
    
    train_config = TrainingConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_ratio=args.train_ratio
    )
    
    net, results, run_dir = train_hebbian_colored_mnist(net_config, train_config) 