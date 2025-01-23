import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np
import scipy.io
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Set publication-quality plot parameters
plt.rcParams.update({
    'font.size': 10,            # Base font size
    'axes.titlesize': 14,       # Title font size
    'axes.labelsize': 12,       # Axis label font size
    'xtick.labelsize': 10,      # X-axis tick label size
    'ytick.labelsize': 10,      # Y-axis tick label size
    'legend.fontsize': 10,      # Legend font size
    'figure.titlesize': 16      # Overall figure title size
})

# Define color labels and RGB values
COLOR_LABELS = ['r', 'b', 'g', 'y', 'c', 'm']  # First letter of each color
COLORS = {
    2: np.array([  # RGB values for 2 colors (easy difficulty)
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
    ]),
    3: np.array([  # RGB values for 3 colors
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
    ]),
    5: np.array([  # RGB values for 5 colors
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [0.0, 1.0, 1.0],  # Cyan
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

def to_device(x, device='gpu'):
    """Move data to specified device."""
    if not HAS_CUDA or device == 'cpu':
        return x.get() if HAS_CUDA and isinstance(x, cp.ndarray) else np.asarray(x)
    return cp.asarray(x) if isinstance(x, np.ndarray) else x

def load_mnist(path='materials/mnist/mnist_all.mat', device='gpu'):
    """Load MNIST data exactly as in KH's implementation."""
    # Load data
    mat = scipy.io.loadmat(path)
    
    # Training data - exactly as in KH
    M = np.zeros((0, 784))
    labels = []
    for i in range(10):
        data = mat[f'train{i}']
        M = np.concatenate((M, data), axis=0)
        labels.extend([i] * len(data))
    M = M / 255.0  # Normalize exactly as in KH
    labels = np.array(labels, dtype=np.int64)
    
    # Test data - same process
    M_test = np.zeros((0, 784))
    labels_test = []
    for i in range(10):
        data = mat[f'test{i}']
        M_test = np.concatenate((M_test, data), axis=0)
        labels_test.extend([i] * len(data))
    M_test = M_test / 255.0
    labels_test = np.array(labels_test, dtype=np.int64)
    
    # Move to device if needed
    if HAS_CUDA and device == 'gpu':
        M = to_device(M, 'gpu')
        M_test = to_device(M_test, 'gpu')
        labels = to_device(labels, 'gpu')
        labels_test = to_device(labels_test, 'gpu')
    
    return M, labels, M_test, labels_test

def get_minibatch(data, labels, batch_size):
    """Get a random minibatch of data."""
    xp = cp if HAS_CUDA and isinstance(data, cp.ndarray) else np
    indices = xp.random.permutation(len(data))[:batch_size]
    return data[indices], labels[indices]

def visualize_weights(net):
    """Visualize network weights.
    
    Args:
        net: HebbianNetwork instance
        
    Returns:
        matplotlib figure
    """
    # Get first layer weights
    W = net.layers[0].W
    if HAS_CUDA:
        W = W.get()
    
    n_units = W.shape[0]  # Number of hidden units
    n_cols = min(10, n_units)
    n_rows = (n_units + n_cols - 1) // n_cols
    
    # Determine if this is colored MNIST based on input size
    input_size = W.shape[1]
    is_colored = input_size > 784
    n_colors = (input_size // 784) - 1 if is_colored else 0
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each unit's weights
    for i in range(n_units):
        row, col = i // n_cols, i % n_cols
        if is_colored:
            # For colored MNIST, show RGB visualization
            intensity = W[i, :784].reshape(28, 28)
            color_channels = [
                W[i, (784 * (c + 1)):(784 * (c + 2))].reshape(28, 28)
                for c in range(n_colors)
            ]
            
            # Create RGB image
            rgb = np.zeros((28, 28, 3))
            colors = COLORS[n_colors]
            for c, channel in enumerate(color_channels):
                for j in range(3):  # RGB channels
                    rgb[:, :, j] += channel * colors[c, j]
            
            # Scale by intensity
            rgb = rgb * intensity.reshape(28, 28, 1)
            
            # Ensure values are in reasonable range
            rgb = np.clip(rgb, -0.5, 0.5)
            
            # Plot
            axes[row, col].imshow(rgb + 0.5)  # Shift to [0, 1] range
        else:
            # For regular MNIST, show grayscale
            weights = W[i].reshape(28, 28)
            axes[row, col].imshow(weights, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_units, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('Hidden Unit Weights')
    plt.tight_layout()
    return fig

def cosine_similarity(v1, v2, xp=np):
    """Compute cosine similarity between two lists of activation vectors."""
    # Stack inputs if they're lists
    if isinstance(v1, list):
        v1 = xp.stack(v1)
    if isinstance(v2, list):
        v2 = xp.stack(v2)
    
    # Compute norms
    norms1 = xp.linalg.norm(v1, axis=1, keepdims=True)
    norms2 = xp.linalg.norm(v2, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = xp.sum(v1 * v2, axis=1) / (norms1.squeeze() * norms2.squeeze() + 1e-8)
    return xp.mean(similarities)

def plot_association_matrix(matrix, save_path, title="Association Matrix", n_colors=None, ood_indices=None):
    """Plot association matrix between signific and referential pathways."""
    if HAS_CUDA and isinstance(matrix, cp.ndarray):
        matrix = matrix.get()
        
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Cosine Similarity', rotation=270, labelpad=15)
    
    # Configure axes
    n_classes = matrix.shape[0]
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    
    # Generate labels with OOD marking
    labels = []
    for i in range(n_classes):
        digit = i % 10
        color = i // 10 if n_colors else None
        if n_colors:
            label = f"{COLOR_LABELS[color]}{digit}"
            if ood_indices is not None and i in ood_indices:
                label += "O"
        else:
            label = str(digit)
        labels.append(label)
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Source Class')
    
    # Add correlation values
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center", 
                         color="white" if abs(matrix[i, j]) > 0.5 else "black",
                         fontsize=8)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_class_averages(data, labels, num_classes, xp=np):
    """Compute average input for each class."""
    averages = []
    for i in range(num_classes):
        mask = labels == i
        class_data = data[mask]
        avg = xp.mean(class_data, axis=0)
        averages.append(avg)
    return xp.stack(averages)

def plot_reconstructions(reconstructions, class_averages, save_path, n_colors=2, ood_indices=None):
    """Plot reconstructions and class averages side by side."""
    # Convert to CPU if needed
    if HAS_CUDA:
        reconstructions = cp.asnumpy(reconstructions)
        class_averages = cp.asnumpy(class_averages)
    
    n_digits = 10
    n_rows = 2 * n_colors  # One row for reconstructions, one for class averages per color
    n_cols = n_digits
    
    # Create figure with extra space on left for row labels and top for title
    fig = plt.figure(figsize=(2*n_cols + 1, 2*n_rows + 1))  # Added height for title
    gs = plt.GridSpec(n_rows, n_cols + 1,  # Add column for labels
                     width_ratios=[0.2] + [1]*n_cols,  # First column narrow for labels
                     top=0.85)  # Leave more space at top for title
    
    # Add main title with larger font
    plt.suptitle('StR Reconstructions and Class Averages', fontsize=20, y=0.95)
    
    # Create mapping from digit-color pairs to indices
    pair_to_idx = {(d, c): i for i, (d, c) in enumerate([(d, c) for d in range(n_digits) for c in range(n_colors)])}
    
    # Plot each color's reconstructions and class averages
    for color in range(n_colors):
        # Row indices for this color
        recon_row = 2 * color  # Row for reconstructions
        avg_row = 2 * color + 1  # Row for class averages
        
        # Add row labels
        recon_label = fig.add_subplot(gs[recon_row, 0])
        recon_label.text(0.5, 0.5, 'Reconstruction', 
                        rotation=90, ha='center', va='center', fontsize=12)
        recon_label.axis('off')
        
        avg_label = fig.add_subplot(gs[avg_row, 0])
        avg_label.text(0.5, 0.5, 'Class Average',
                      rotation=90, ha='center', va='center', fontsize=12)
        avg_label.axis('off')
        
        # Plot each digit for this color
        for digit in range(n_digits):
            idx = pair_to_idx[(digit, color)]
            
            # Skip if this combination is not in our data
            if idx >= len(reconstructions) or idx >= len(class_averages):
                continue
            
            # Plot reconstruction
            ax = fig.add_subplot(gs[recon_row, digit + 1])  # +1 for label column
            recon_rgb = multichannel_to_rgb(reconstructions[idx], n_colors)
            if HAS_CUDA:
                recon_rgb = cp.asnumpy(recon_rgb)
            ax.imshow(recon_rgb)
            ax.axis('off')
            
            # Add label with OOD marker if needed
            label = f"{COLOR_LABELS[color]}{digit}"
            if ood_indices and idx in ood_indices:
                label += "O"
            ax.set_title(label, fontsize=14)  # Increased font size for image labels
            
            # Plot class average
            ax = fig.add_subplot(gs[avg_row, digit + 1])  # +1 for label column
            avg_rgb = multichannel_to_rgb(class_averages[idx], n_colors)
            if HAS_CUDA:
                avg_rgb = cp.asnumpy(avg_rgb)
            ax.imshow(avg_rgb)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path, title=None, normalize=False, is_color=False, n_colors=None, is_compositional=False, ood_indices=None):
    """Plot confusion matrix with publication-quality styling."""
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    n_classes = conf_matrix.shape[0]
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    
    # Use a perceptually uniform colormap
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='equal')
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Normalized Count' if normalize else 'Count', 
                      rotation=270, labelpad=15)
    
    # Configure axes
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    
    # Generate labels based on type
    if is_compositional:
        # For compositional classes (e.g. 'r9' for red 9)
        labels = []
        for i in range(n_classes):
            digit = i % 10
            color = i // 10
            label = f"{COLOR_LABELS[color]}{digit}"
            if ood_indices is not None and i in ood_indices:
                label += "O"
            labels.append(label)
    elif is_color:
        # For color-only plots (e.g. 'r' for red)
        labels = [COLOR_LABELS[i] for i in range(n_colors)]
    else:
        # For digit-only plots (e.g. '9' for nine)
        labels = [str(i) for i in range(n_classes)]
    
    # Rotate x-axis labels for better readability in compositional case
    if is_compositional:
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    else:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    # Add counts as text with smaller font size for compositional case
    thresh = conf_matrix.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            fontsize = 8 if is_compositional else 10  # Match str_rec_similarity_matrix font size
            text = ax.text(j, i, format(conf_matrix[i, j], fmt),
                         ha="center", va="center",
                         color="white" if conf_matrix[i, j] > thresh else "black",
                         fontsize=fontsize)
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_similarity_matrix(reconstructions, class_averages, save_path, n_colors=None, ood_indices=None):
    """Plot similarity matrix between reconstructions and class averages using SSIM."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
    
    # Convert to RGB if colored MNIST
    if n_colors is not None:
        recon_images = [multichannel_to_rgb(r, n_colors) for r in reconstructions]
        avg_images = [multichannel_to_rgb(a, n_colors) for a in class_averages]
        channel_axis = -1  # RGB channels are in the last dimension
    else:
        recon_images = [r.reshape(28, 28) for r in reconstructions]
        avg_images = [a.reshape(28, 28) for a in class_averages]
        channel_axis = None  # No color channels
    
    # Compute similarity matrix using SSIM
    n_samples = len(recon_images)
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        recon_i = recon_images[i]
        for j in range(n_samples):
            avg_j = avg_images[j]
            similarity_matrix[i, j] = ssim(recon_i, avg_j, data_range=1.0, channel_axis=channel_axis)
    
    # Create normalized version
    normalized_matrix = similarity_matrix.copy()
    for i in range(n_samples):
        row_min = normalized_matrix[i].min()
        row_max = normalized_matrix[i].max()
        if row_max > row_min:
            normalized_matrix[i] = (normalized_matrix[i] - row_min) / (row_max - row_min)
    
    # Plot both matrices
    for matrix, suffix in [(similarity_matrix, ''), (normalized_matrix, '_normalized')]:
        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        vmin, vmax = (-1, 1) if suffix == '' else (0, 1)
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Add colorbar with scientific styling
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('SSIM' + (' (Normalized)' if suffix == '_normalized' else ''), 
                          rotation=270, labelpad=15)
        
        # Configure axes
        ax.set_xticks(np.arange(n_samples))
        ax.set_yticks(np.arange(n_samples))
        
        # Generate labels
        labels = []
        for i in range(n_samples):
            digit = i % 10
            color = i // 10 if n_colors else None
            if n_colors:
                label = f"{COLOR_LABELS[color]}{digit}"
                if ood_indices is not None and i in ood_indices:
                    label += "O"
            else:
                label = str(digit)
            labels.append(label)
        
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Class Average')
        ax.set_ylabel('Reconstruction')
        
        # Add similarity values
        for i in range(n_samples):
            for j in range(n_samples):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                             ha="center", va="center",
                             color="black" if abs(matrix[i, j]) < 0.5 else "white",
                             fontsize=8)
        
        plt.title("StR-rec Similarity Matrix" + (' (Normalized)' if suffix == '_normalized' else ''))
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', f'{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compute_compositional_confusion_matrix(digit_preds, color_preds, digit_labels, color_labels):
    """Compute confusion matrix for compositional predictions."""
    n_colors = len(np.unique(color_labels))
    n_digits = 10
    n_classes = n_colors * n_digits
    
    # Convert to compositional class indices
    true_classes = color_labels * n_digits + digit_labels
    pred_classes = color_preds * n_digits + digit_preds
    
    # Compute confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes))
    for i in range(len(true_classes)):
        conf_matrix[true_classes[i], pred_classes[i]] += 1
    
    return conf_matrix

def compute_confusion_matrix(true_labels, pred_labels, n_classes=None):
    """Compute confusion matrix between true and predicted labels.
    
    Args:
        true_labels: Array of true labels
        pred_labels: Array of predicted labels
        n_classes: Number of classes. If None, inferred from labels.
        
    Returns:
        confusion_matrix: Array of shape (n_classes, n_classes)
    """
    # Move to CPU if needed
    if HAS_CUDA:
        if isinstance(true_labels, cp.ndarray):
            true_labels = true_labels.get()
        if isinstance(pred_labels, cp.ndarray):
            pred_labels = pred_labels.get()
    
    # Convert to integer labels
    true_labels = true_labels.astype(int)
    pred_labels = pred_labels.astype(int)
    
    if n_classes is None:
        n_classes = max(np.max(true_labels), np.max(pred_labels)) + 1
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for t, p in zip(true_labels, pred_labels):
        confusion_matrix[t, p] += 1
    
    return confusion_matrix 