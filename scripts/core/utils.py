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

# Define RGB values for each color
COLORS = {
    2: np.array([  # RGB values for 2 colors (easy difficulty)
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
    ]),
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

def plot_association_matrix(matrix, save_path, title="Association Matrix"):
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
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Referential Class')
    ax.set_ylabel('Signific Class')
    
    # Add correlation values
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center", 
                         color="black" if abs(matrix[i, j]) < 0.5 else "white",
                         fontsize=8)
    
    plt.title(title)
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

def plot_reconstructions(reconstructions, class_averages, save_path, n_colors=None):
    """Plot reconstructed MNIST digits from signific inputs.
    
    Args:
        reconstructions: Array of reconstructed inputs
        class_averages: Array of class average inputs
        save_path: Path to save the plot
        n_colors: Number of colors (None for regular MNIST)
    """
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
    
    # Determine if this is colored MNIST
    if n_colors is None:
        # Regular MNIST - reshape to 28x28
        recon_images = [r.reshape(28, 28) for r in reconstructions]
        avg_images = [a.reshape(28, 28) for a in class_averages]
        cmap = 'gray'
        n_rows = 2  # Just reconstruction and class average rows
    else:
        # Colored MNIST - convert to RGB
        recon_images = [multichannel_to_rgb(r, n_colors) for r in reconstructions]
        avg_images = [multichannel_to_rgb(a, n_colors) for a in class_averages]
        cmap = None  # Don't use colormap for RGB images
        n_rows = 2 * n_colors  # One row per color for both recon and avg
    
    n_samples = len(recon_images) // n_colors
    n_cols = 10  # Always 10 digits
    
    # Create figure and axes grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Add overall title
    plt.suptitle('StR Reconstructions', y=1.0)
    
    # Plot reconstructions and class averages by color
    for color in range(n_colors):
        # Plot reconstructions for this color
        recon_row = color * 2  # Even rows for reconstructions
        for digit in range(10):
            idx = color * 10 + digit
            if idx < len(recon_images):
                axes[recon_row, digit].imshow(recon_images[idx], cmap=cmap, vmin=0, vmax=1)
                axes[recon_row, digit].axis('off')
                axes[recon_row, digit].set_title(f'd:{digit}')
        axes[recon_row, 0].set_ylabel(f'Recon\nColor {color}', rotation=0, ha='right', va='center')
        
        # Plot class averages for this color
        avg_row = color * 2 + 1  # Odd rows for class averages
        for digit in range(10):
            idx = color * 10 + digit
            if idx < len(avg_images):
                axes[avg_row, digit].imshow(avg_images[idx], cmap=cmap, vmin=0, vmax=1)
                axes[avg_row, digit].axis('off')
        axes[avg_row, 0].set_ylabel(f'Avg\nColor {color}', rotation=0, ha='right', va='center')
    
    # Adjust spacing
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path, title=None, normalize=False):
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
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    # Add counts as text
    thresh = conf_matrix.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, format(conf_matrix[i, j], fmt),
                         ha="center", va="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_similarity_matrix(reconstructions, class_averages, save_path, n_colors=None):
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
    
    # Plot matrix
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('SSIM', rotation=270, labelpad=15)
    
    # Configure axes
    ax.set_xticks(np.arange(n_samples))
    ax.set_yticks(np.arange(n_samples))
    ax.set_xlabel('Class Average')
    ax.set_ylabel('Reconstruction')
    
    # Add similarity values
    for i in range(n_samples):
        for j in range(n_samples):
            text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                         ha="center", va="center",
                         color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white",
                         fontsize=8)
    
    plt.title("StR-rec Similarity Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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