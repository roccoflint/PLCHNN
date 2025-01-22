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

from experiments.colored_mnist import (
    generate_colored_mnist, generate_holdout_splits,
    multichannel_to_rgb, visualize_colored_digits,
    train_hebbian_colored_mnist
)

def test_dataset_generation():
    """Test basic dataset generation."""
    n_colors = 3
    train_data, train_labels, test_data, test_labels = generate_colored_mnist(n_colors=n_colors)
    
    # Check shapes
    assert train_data.shape[1] == 784 * (1 + n_colors), "Wrong feature dimension"
    assert train_labels.shape[1] == 2 + 10 + n_colors, "Wrong label dimension"  # [digit, color, signific]
    assert test_data.shape[1] == 784 * (1 + n_colors), "Wrong feature dimension"
    assert test_labels.shape[1] == 2 + 10 + n_colors, "Wrong label dimension"  # [digit, color, signific]
    
    # Check label ranges
    assert np.all(train_labels[:, 0] >= 0) and np.all(train_labels[:, 0] < 10), "Invalid digit labels"
    assert np.all(train_labels[:, 1] >= 0) and np.all(train_labels[:, 1] < n_colors), "Invalid color labels"
    
    # Check signific vectors
    for i in range(len(train_labels)):
        digit = int(train_labels[i, 0])
        color = int(train_labels[i, 1])
        signific = train_labels[i, 2:]
        
        # Check it's a 2-hot vector
        assert np.sum(signific) == 2, "Signific vector should be 2-hot"
        
        # Check digit encoding
        assert signific[digit] == 1, "Wrong digit encoding"
        assert np.sum(signific[:10]) == 1, "Multiple digits encoded"
        
        # Check color encoding
        assert signific[10 + color] == 1, "Wrong color encoding"
        assert np.sum(signific[10:]) == 1, "Multiple colors encoded"
        
        # Only check first few samples
        if i >= 2:
            break
    
    # Check color channel consistency
    for i in range(len(train_data)):
        intensity = train_data[i, :784]
        assigned_color = int(train_labels[i, 1])
        
        # Get all color channels
        color_channels = [
            train_data[i, (784 * (j + 1)):(784 * (j + 2))]
            for j in range(n_colors)
        ]
        
        # Check that assigned color matches intensity pattern
        assigned_channel = color_channels[assigned_color]
        assert np.all((assigned_channel > 0) == (intensity > 0)), \
            "Assigned color doesn't match intensity pattern"
        
        # Check other channels are zero
        for j, channel in enumerate(color_channels):
            if j != assigned_color:
                assert np.all(channel == 0), f"Non-assigned color channel {j} is nonzero"
        
        # Only check first few samples
        if i >= 2:
            break
    
    print("Dataset generation tests passed!")

def test_holdout_splits():
    """Test holdout split generation."""
    n_colors = 3
    train_data, train_labels, test_data, test_labels = generate_colored_mnist(n_colors=n_colors)
    
    # Define holdout pairs
    holdout_pairs = [(0, 0), (1, 1)]  # Hold out digit 0-color 0 and digit 1-color 1
    
    # Generate splits
    train_split_data, train_split_labels, test_split_data, test_split_labels = \
        generate_holdout_splits(train_data, train_labels, holdout_pairs)
    
    # Check that holdout pairs only appear in test set
    for digit, color in holdout_pairs:
        train_mask = (train_split_labels[:, 0] == digit) & (train_split_labels[:, 1] == color)
        assert not np.any(train_mask), f"Holdout pair ({digit}, {color}) found in training set"
        
        test_mask = (test_split_labels[:, 0] == digit) & (test_split_labels[:, 1] == color)
        assert np.any(test_mask), f"Holdout pair ({digit}, {color}) not found in test set"
        
        # Check corresponding signific vectors are also held out
        if len(train_split_labels.shape) > 2:  # If signific vectors included
            train_signific = train_split_labels[:, 2:]
            test_signific = test_split_labels[:, 2:]
            
            # Check no training examples have this digit-color combination in signific
            train_digit_mask = train_signific[:, digit] == 1
            train_color_mask = train_signific[:, 10 + color] == 1
            assert not np.any(train_digit_mask & train_color_mask), \
                f"Holdout signific pair ({digit}, {color}) found in training set"
            
            # Check test set has this digit-color combination in signific
            test_digit_mask = test_signific[:, digit] == 1
            test_color_mask = test_signific[:, 10 + color] == 1
            assert np.any(test_digit_mask & test_color_mask), \
                f"Holdout signific pair ({digit}, {color}) not found in test set"
    
    print("Holdout split tests passed!")

def test_visualization():
    """Test visualization functions."""
    n_colors = 3
    train_data, train_labels, _, _ = generate_colored_mnist(n_colors=n_colors)
    
    # Test single image conversion
    rgb = multichannel_to_rgb(train_data[0], n_colors)
    assert rgb.shape == (28, 28, 3), "Wrong RGB shape for single image"
    assert np.all(rgb >= 0) and np.all(rgb <= 1), "RGB values out of range"
    
    # Test batch conversion
    batch_rgb = multichannel_to_rgb(train_data[:5], n_colors)
    assert batch_rgb.shape == (5, 28, 28, 3), "Wrong RGB shape for batch"
    assert np.all(batch_rgb >= 0) and np.all(batch_rgb <= 1), "RGB values out of range"
    
    # Test visualization
    fig = visualize_colored_digits(train_data[:10], train_labels[:10], n_colors)
    assert fig is not None, "Visualization failed"
    
    print("Visualization tests passed!")

def test_training():
    """Test training on colored MNIST with minimal settings."""
    # Train network with minimal settings
    net, results, run_dir = train_hebbian_colored_mnist(
        n_colors=2,  # Easy difficulty
        hidden_size=50,  # Small network for quick testing
        n_epochs=1,  # Single epoch
        batch_size=100
    )
    
    # Check that results contain expected metrics
    expected_metrics = [
        'train_digit_acc_decoder', 'train_color_acc_decoder', 'train_full_acc_decoder',
        'test_digit_acc_decoder', 'test_color_acc_decoder', 'test_full_acc_decoder',
        'train_digit_acc_rts', 'train_color_acc_rts', 'train_full_acc_rts',
        'test_digit_acc_rts', 'test_color_acc_rts', 'test_full_acc_rts'
    ]
    for metric in expected_metrics:
        assert metric in results, f"Missing {metric} in results"
        value = float(results[metric])  # Convert to scalar
        assert 0 <= value <= 1, f"Invalid accuracy range for {metric}: {value}"
    
    # Check that files were saved
    assert os.path.exists(run_dir), "Run directory not created"
    assert os.path.exists(os.path.join(run_dir, 'config.txt')), "Config not saved"
    assert os.path.exists(os.path.join(run_dir, 'summary.txt')), "Summary not saved"
    assert os.path.exists(os.path.join(run_dir, 'results.npy')), "Results not saved"
    
    print("Training tests passed!")

if __name__ == "__main__":
    print("Running colored MNIST tests...")
    test_dataset_generation()
    test_holdout_splits()
    test_visualization()
    test_training()
    print("All tests passed!") 