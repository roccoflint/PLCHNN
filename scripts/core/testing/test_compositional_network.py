import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np

from core.network import HebbianNetwork
from experiments.colored_mnist import generate_colored_mnist, generate_holdout_splits, compute_compositional_averages, compute_component_averages

def test_compositional_network():
    """Test network with compositional signific pathway."""
    # Generate small colored MNIST dataset
    n_colors = 3
    train_data, train_labels, test_data, test_labels = generate_colored_mnist(
        n_colors=n_colors, device='cpu'
    )
    
    # Extract signific vectors
    train_signific = train_labels[:, 2:]  # Shape: (n_samples, 10 + n_colors)
    
    # Create network
    layer_sizes = [784 * (1 + n_colors), 100]  # Single hidden layer for simplicity
    signific_size = 10 + n_colors  # 10 digits + n_colors
    net = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],  # Connect to hidden layer
        p=2.0,
        delta=0.4,
        k=2,
        device='cpu'
    )
    
    # Test forward pass with no pathway interaction (training mode)
    batch_size = 10
    batch_data = train_data[:batch_size]
    batch_signific = train_signific[:batch_size]
    
    # During training, pathways should not interact
    h = net.forward(batch_data, batch_signific, allow_pathway_interaction=False)
    assert h.shape == (signific_size, batch_size), "Wrong output shape"
    
    # Test RtS classification
    component_sizes = [10, n_colors]  # [digits, colors]
    classifications = net.rts_classify(batch_data, component_sizes)
    assert len(classifications) == 2, "Should return digit and color classifications"
    assert classifications[0].shape == (batch_size,), "Wrong digit classification shape"
    assert classifications[1].shape == (batch_size,), "Wrong color classification shape"
    
    # Test weight updates
    net.update(batch_data, batch_signific, learning_rate=0.02)
    
    print("Compositional network tests passed!")

def test_pathway_separation():
    """Test that pathways don't influence each other during training."""
    # Generate small dataset
    n_colors = 3
    train_data, train_labels, _, _ = generate_colored_mnist(
        n_colors=n_colors, device='cpu'
    )
    train_signific = train_labels[:, 2:]
    
    # Create two networks - one with pathway interaction, one without
    layer_sizes = [784 * (1 + n_colors), 100]
    signific_size = 10 + n_colors
    
    net_no_interact = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],
        allow_pathway_interaction=False,
        device='cpu'
    )
    
    net_interact = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],
        allow_pathway_interaction=True,
        device='cpu'
    )
    
    # Copy weights to ensure same initialization
    net_interact.layers[0].W = net_no_interact.layers[0].W.copy()
    net_interact.signific_weights[0] = net_no_interact.signific_weights[0].copy()
    
    # Forward pass on same batch
    batch_size = 10
    batch_data = train_data[:batch_size]
    batch_signific = train_signific[:batch_size]
    
    # Get hidden activations
    h_no_interact = net_no_interact._get_hidden_activations(batch_data)
    h_interact = net_interact._get_hidden_activations(batch_data)
    
    # Without pathway interaction, activations should be identical
    assert np.allclose(h_no_interact[0], h_interact[0]), \
        "Pathways affecting each other during training"
    
    print("Pathway separation tests passed!")

def test_compositional_generalization():
    """Test RtS evaluation on compositional generalization.
    
    Tests that network can:
    1. Classify digits and colors separately
    2. Evaluate performance on held-out compositions
    3. Compute per-component accuracy metrics
    """
    # Generate dataset
    n_colors = 3
    train_data, train_labels, test_data, test_labels = generate_colored_mnist(
        n_colors=n_colors, device='cpu'
    )
    
    # Create holdout splits
    holdout_pairs = [(0, 0), (1, 1)]  # Hold out digit 0-color 0 and digit 1-color 1
    train_split_data, train_split_labels, test_split_data, test_split_labels = \
        generate_holdout_splits(train_data, train_labels, holdout_pairs)
    
    # Create and train network (minimal training for test)
    layer_sizes = [784 * (1 + n_colors), 100]
    signific_size = 10 + n_colors
    net = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],
        device='cpu'
    )
    
    # Train on a small batch
    train_batch_size = 100
    train_batch_data = train_split_data[:train_batch_size]
    train_batch_signific = train_split_labels[:, 2:][:train_batch_size]
    net.update(train_batch_data, train_batch_signific, learning_rate=0.02)
    
    # Find indices of holdout pairs in test set
    test_labels_np = test_split_labels[:, :2]
    holdout_mask = np.zeros(len(test_labels_np), dtype=bool)
    for digit, color in holdout_pairs:
        pair_mask = (test_labels_np[:, 0] == digit) & (test_labels_np[:, 1] == color)
        holdout_mask = holdout_mask | pair_mask
    
    # Get indices of some holdout and non-holdout samples
    holdout_indices = np.where(holdout_mask)[0][:10]  # Get first 10 holdout samples
    non_holdout_indices = np.where(~holdout_mask)[0][:90]  # Get 90 non-holdout samples
    test_indices = np.concatenate([holdout_indices, non_holdout_indices])
    np.random.shuffle(test_indices)
    
    # Create test batch with mix of holdout and non-holdout samples
    test_batch_data = test_split_data[test_indices]
    test_batch_labels = test_split_labels[test_indices]
    holdout_labels = test_batch_labels[:, :2]  # Just digit and color labels
    
    # Get RtS classifications
    component_sizes = [10, n_colors]  # [digits, colors]
    digit_pred, color_pred = net.rts_classify(test_batch_data, component_sizes)
    
    # Convert to numpy for evaluation
    if HAS_CUDA:
        digit_pred = cp.asnumpy(digit_pred)
        color_pred = cp.asnumpy(color_pred)
        holdout_labels = cp.asnumpy(holdout_labels)
    
    # Compute accuracies
    digit_acc = np.mean(digit_pred == holdout_labels[:, 0])
    color_acc = np.mean(color_pred == holdout_labels[:, 1])
    
    # Compute accuracies specifically for holdout pairs
    holdout_mask = np.zeros(len(holdout_labels), dtype=bool)
    for digit, color in holdout_pairs:
        pair_mask = (holdout_labels[:, 0] == digit) & (holdout_labels[:, 1] == color)
        holdout_mask = holdout_mask | pair_mask
    
    # Only compute holdout accuracies if we have holdout samples
    if np.any(holdout_mask):
        holdout_digit_acc = np.mean(digit_pred[holdout_mask] == holdout_labels[holdout_mask, 0])
        holdout_color_acc = np.mean(color_pred[holdout_mask] == holdout_labels[holdout_mask, 1])
    else:
        print("Warning: No holdout samples found in batch")
        holdout_digit_acc = 0.0
        holdout_color_acc = 0.0
    
    # Store results (would normally save these)
    results = {
        'digit_acc': float(digit_acc),
        'color_acc': float(color_acc),
        'holdout_digit_acc': float(holdout_digit_acc),
        'holdout_color_acc': float(holdout_color_acc),
        'n_holdout_samples': int(np.sum(holdout_mask))
    }
    
    # Basic sanity checks
    assert 0 <= results['digit_acc'] <= 1, "Invalid digit accuracy"
    assert 0 <= results['color_acc'] <= 1, "Invalid color accuracy"
    assert 0 <= results['holdout_digit_acc'] <= 1, "Invalid holdout digit accuracy"
    assert 0 <= results['holdout_color_acc'] <= 1, "Invalid holdout color accuracy"
    assert results['n_holdout_samples'] > 0, "No holdout samples in test batch"
    
    print("Compositional generalization tests passed!")
    return results

def test_class_averages():
    """Test computation of class averages for colored MNIST."""
    # Generate small dataset
    n_colors = 3
    train_data, train_labels, _, _ = generate_colored_mnist(
        n_colors=n_colors, device='cpu'
    )
    base_labels = train_labels[:, :2]  # Just digit and color labels
    
    # Compute compositional averages
    comp_averages = compute_compositional_averages(train_data, base_labels, n_colors)
    
    # Check compositional averages
    for digit in range(10):
        for color in range(n_colors):
            # Find samples with this combination
            mask = (base_labels[:, 0] == digit) & (base_labels[:, 1] == color)
            if np.any(mask):
                # Should have this average
                assert (digit, color) in comp_averages, f"Missing average for digit {digit}, color {color}"
                
                # Check shape
                avg = comp_averages[(digit, color)]
                assert avg.shape == (784 * (1 + n_colors),), "Wrong average shape"
                
                # Check values are reasonable
                assert np.all(avg >= 0), "Negative values in average"
                assert np.all(avg <= 1), "Values > 1 in average"
                
                # Check color channel consistency
                intensity = avg[:784]
                color_channels = [
                    avg[(784 * (i + 1)):(784 * (i + 2))]
                    for i in range(n_colors)
                ]
                
                # Only assigned color should be active where intensity is nonzero
                for i, channel in enumerate(color_channels):
                    if i == color:
                        assert np.allclose((channel > 0), (intensity > 0)), \
                            f"Wrong activation pattern for color {color}"
                    else:
                        assert np.allclose(channel, 0), \
                            f"Non-zero values in wrong color channel {i}"
    
    # Compute component averages
    digit_averages, color_averages = compute_component_averages(train_data, base_labels, n_colors)
    
    # Check digit averages
    for digit in range(10):
        if digit in digit_averages:
            avg = digit_averages[digit]
            assert avg.shape == (784 * (1 + n_colors),), "Wrong digit average shape"
            assert np.all(avg >= 0), "Negative values in digit average"
            assert np.all(avg <= 1), "Values > 1 in digit average"
    
    # Check color averages
    for color in range(n_colors):
        if color in color_averages:
            avg = color_averages[color]
            assert avg.shape == (784 * (1 + n_colors),), "Wrong color average shape"
            assert np.all(avg >= 0), "Negative values in color average"
            assert np.all(avg <= 1), "Values > 1 in color average"
    
    print("Class average tests passed!")

def test_compositional_reconstruction():
    """Test StR reconstruction with compositional inputs."""
    # Generate small dataset
    n_colors = 3
    train_data, train_labels, _, _ = generate_colored_mnist(
        n_colors=n_colors, device='cpu'
    )
    base_labels = train_labels[:, :2]  # Just digit and color labels
    
    # Create network
    layer_sizes = [784 * (1 + n_colors), 100]
    signific_size = 10 + n_colors
    net = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=[True],
        device='cpu'
    )
    
    # Train on a small batch
    train_batch_size = 100
    train_batch_data = train_data[:train_batch_size]
    train_batch_signific = train_labels[:, 2:][:train_batch_size]
    net.update(train_batch_data, train_batch_signific, learning_rate=0.02)
    
    # Create test signific input for digit 0, color 1
    signific_input = np.zeros((1, signific_size))
    signific_input[0, 0] = 1  # Digit 0
    signific_input[0, 10 + 1] = 1  # Color 1
    
    # Get reconstruction with component-wise optimization
    component_sizes = [10, n_colors]  # [digits, colors]
    recon = net.str_rec(signific_input, max_iter=100, component_sizes=component_sizes)
    
    # Basic checks
    assert recon.shape == (1, 784 * (1 + n_colors)), "Wrong reconstruction shape"
    assert np.all(recon >= 0) and np.all(recon <= 1), "Values outside [0,1] range"
    
    # Check color channel structure
    intensity = recon[0, :784]
    color_channels = [
        recon[0, (784 * (i + 1)):(784 * (i + 2))]
        for i in range(n_colors)
    ]
    
    # Color 1 should be active where intensity is nonzero
    nonzero_mask = intensity > 0
    assert np.any(color_channels[1][nonzero_mask] > 0), "Target color not active"
    
    # Other colors should be less active
    other_colors = [0, 2]
    for c in other_colors:
        assert np.mean(color_channels[c][nonzero_mask]) < np.mean(color_channels[1][nonzero_mask]), \
            f"Color {c} more active than target color 1"
    
    print("Compositional reconstruction tests passed!")

if __name__ == "__main__":
    print("Running compositional network tests...")
    test_compositional_network()
    test_pathway_separation()
    results = test_compositional_generalization()
    test_class_averages()
    test_compositional_reconstruction()
    print("\nEvaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("\nAll tests passed!") 