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
import matplotlib.gridspec as gridspec

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

class ColoredMNIST:
    """Handles colored MNIST data representation and transformations."""
    
    # Default color mappings
    DEFAULT_COLORS = {
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
    
    COLOR_LABELS = ['r', 'b', 'g', 'y', 'c']  # First letter of each color
    
    def __init__(self, n_colors, device='cpu'):
        """Initialize ColoredMNIST handler.
        
        Args:
            n_colors: Number of colors to use
            device: Device to store data on ('cpu' or 'gpu')
        """
        self.n_colors = n_colors
        self.device = device
        self.colors = self._to_device(self.DEFAULT_COLORS[n_colors])
    
    def _to_device(self, x):
        """Move data to specified device."""
        if not HAS_CUDA or self.device == 'cpu':
            return x.get() if HAS_CUDA and isinstance(x, cp.ndarray) else np.asarray(x)
        return cp.asarray(x) if isinstance(x, np.ndarray) else x
    
    def _get_array_module(self, x):
        """Get the appropriate array module (numpy or cupy) for the input."""
        return cp if HAS_CUDA and isinstance(x, cp.ndarray) else np
    
    def to_rgb(self, data):
        """Convert multi-channel representation to RGB image.
        
        Args:
            data: Array of shape (784 * (1 + n_colors),) or (batch_size, 784 * (1 + n_colors))
            
        Returns:
            Array of shape (28, 28, 3) or (batch_size, 28, 28, 3) containing RGB values
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        batch_size = data.shape[0]
        xp = self._get_array_module(data)
        
        # Extract intensity and color channels
        intensity = data[:, :784].reshape(batch_size, 28, 28)
        color_channels = [
            data[:, (784 * (i + 1)):(784 * (i + 2))].reshape(batch_size, 28, 28)
            for i in range(self.n_colors)
        ]
        
        # Initialize RGB output
        rgb = xp.zeros((batch_size, 28, 28, 3))
        
        # Combine channels using color mapping
        for i, color_channel in enumerate(color_channels):
            for j in range(3):  # RGB channels
                rgb[:, :, :, j] += color_channel * self.colors[i, j]
        
        # Scale by intensity
        rgb = rgb * intensity.reshape(batch_size, 28, 28, 1)
        
        # Ensure values are in [0, 1]
        rgb = xp.clip(rgb, 0, 1)
        
        return rgb[0] if data.shape[0] == 1 else rgb
    
    def get_label(self, digit, color, mark_ood=False):
        """Get string label for a digit-color combination.
        
        Args:
            digit: Digit label (0-9)
            color: Color index
            mark_ood: Whether to mark out-of-distribution samples
            
        Returns:
            String label (e.g. 'r5' for red 5, 'r5O' for OOD red 5)
        """
        label = f"{self.COLOR_LABELS[color]}{digit}"
        if mark_ood:
            label = f"{label}O"
        return label
    
    def plot_samples(self, data, labels, n_samples=10, title=None):
        """Plot colored MNIST samples.
        
        Args:
            data: Array of colored MNIST data
            labels: Array of [digit, color] labels
            n_samples: Number of samples to plot
            title: Plot title
            
        Returns:
            matplotlib figure
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
            rgb = self.to_rgb(data[idx])
            axes[row, col].imshow(rgb)
            digit, color = labels[idx, 0], labels[idx, 1]
            label = self.get_label(digit, color)
            axes[row, col].set_title(label)
            axes[row, col].axis('off')
        
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        return fig

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

def plot_association_matrix(matrix, save_path, title="Association Matrix", n_colors=None, train_indices=None):
    """Plot an association matrix with proper labels and colorbar."""
    plt.close()
    
    # Initialize ColoredMNIST handler if needed
    mnist = ColoredMNIST(n_colors) if n_colors is not None else None
    
    # Determine matrix type from save path
    is_id = 'id' in save_path
    is_ood = 'ood' in save_path
    
    # For ID/OOD splits, filter matrix to only show relevant combinations
    if is_id or is_ood:
        # Create mask for rows/columns
        mask = np.zeros(matrix.shape[0], dtype=bool)
        for i in range(matrix.shape[0]):
            in_train = train_indices is not None and i in train_indices
            mask[i] = (is_id and in_train) or (is_ood and not in_train)
        
        # Filter matrix
        matrix = matrix[mask][:, mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='viridis')
    plt.colorbar(im)
    
    # Generate labels
    labels = []
    for i in range(matrix.shape[0]):
        if mnist is not None:
            digit = i % 10
            color = i // 10
            is_ood = train_indices is not None and i not in train_indices
            label = mnist.get_label(digit, color, mark_ood=is_ood)
        else:
            label = str(i)
        labels.append(label)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add title
    if title == "Association Matrix":
        title = "StR-rep Association Matrix"
        if is_id:
            title += " - ID Only"
        elif is_ood:
            title += " - OOD Only"
    plt.title(title, fontsize=16)
    
    # Save with high DPI
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

def plot_reconstructions(reconstructions, class_averages, save_path, n_colors=2, train_indices=None):
    """Plot reconstructions and class averages."""
    plt.close('all')
    
    # Initialize ColoredMNIST handler
    mnist = ColoredMNIST(n_colors)
    
    # Calculate dimensions
    n_digits = 10
    
    # Create figure with subplots for each color
    fig = plt.figure(figsize=(15, 2*n_colors))
    
    # Create GridSpec with proper spacing
    gs = gridspec.GridSpec(2*n_colors, n_digits + 1, width_ratios=[0.5] + [1]*n_digits)
    gs.update(wspace=0.2, hspace=0.3)  # Reduce horizontal spacing
    
    # Plot reconstructions and class averages for each color
    for color in range(n_colors):
        # Add row labels in their own column
        recon_label = plt.subplot(gs[2*color, 0])
        recon_label.text(1, 0.5, f"{COLOR_LABELS[color]} reconstructions", 
                        rotation=0, ha='right', va='center')
        recon_label.axis('off')
        
        avg_label = plt.subplot(gs[2*color + 1, 0])
        avg_label.text(1, 0.5, f"{COLOR_LABELS[color]} class averages",
                      rotation=0, ha='right', va='center')
        avg_label.axis('off')
        
        # Plot reconstructions and class averages
        for digit in range(n_digits):
            idx = color * n_digits + digit
            
            # Plot reconstruction
            ax = plt.subplot(gs[2*color, digit + 1])
            img_rgb = mnist.to_rgb(reconstructions[idx])
            ax.imshow(img_rgb)
            ax.axis('off')
            
            # Add label with OOD marker if needed
            is_ood = train_indices is not None and idx not in train_indices
            label = mnist.get_label(digit, color, mark_ood=is_ood)
            ax.set_title(label, fontsize=10, pad=2)  # Reduce padding
            
            # Plot class average
            ax = plt.subplot(gs[2*color + 1, digit + 1])
            img_rgb = mnist.to_rgb(class_averages[idx])
            ax.imshow(img_rgb)
            ax.axis('off')
            
            # Add label with OOD marker if needed
            label = mnist.get_label(digit, color, mark_ood=is_ood)
            ax.set_title(label, fontsize=10, pad=2)  # Reduce padding
    
    plt.suptitle("StR-rec Reconstructions", fontsize=16, y=1.02)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path, title=None, normalize=False, is_color=False, n_colors=None, is_compositional=False, train_indices=None):
    """Plot confusion matrix with proper labels and normalization."""
    plt.close()
    
    # Initialize ColoredMNIST handler if needed
    mnist = ColoredMNIST(n_colors) if n_colors is not None else None
    
    # Determine matrix type from save path
    is_training = 'train' in save_path
    is_ood = 'ood' in save_path
    is_id = 'id' in save_path
    
    # For ID/OOD splits, filter matrix to only show relevant combinations
    if is_compositional and (is_id or is_ood):
        # Create mask for rows/columns
        mask = np.zeros(conf_matrix.shape[0], dtype=bool)
        for i in range(conf_matrix.shape[0]):
            in_train = train_indices is not None and i in train_indices
            mask[i] = (is_id and in_train) or (is_ood and not in_train)
        
        # Filter matrix to only include relevant combinations
        filtered_matrix = conf_matrix[mask][:, mask]
        
        # Only use filtered matrix if it's not empty
        if filtered_matrix.size > 0:
            conf_matrix = filtered_matrix
    
    # Plot matrix
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.where(row_sums > 0, conf_matrix.astype('float') / row_sums, 0)
    
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    
    # Generate labels based on matrix type
    if is_color:
        labels = [COLOR_LABELS[i] for i in range(n_colors)]
    elif is_compositional:
        labels = []
        for i in range(conf_matrix.shape[0]):
            digit = i % 10
            color = i // 10
            is_ood = train_indices is not None and i not in train_indices
            label = mnist.get_label(digit, color, mark_ood=is_ood)
            labels.append(label)
    else:
        labels = [str(i) for i in range(conf_matrix.shape[0])]
    
    # Set labels
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    # Add counts as text
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            text = f'{conf_matrix[i, j]:.2f}' if normalize else str(int(conf_matrix[i, j]))
            plt.text(j, i, text, ha='center', va='center')
    
    # Set title
    if title is None:
        title = f'{"Training" if is_training else "Test"} Confusion Matrix'
        if is_color:
            title += ' (Colors)'
        elif is_compositional:
            title += ' (Compositional)'
            if is_id:
                title += ' - ID Only'
            elif is_ood:
                title += ' - OOD Only'
        else:
            title += ' (Digits)'
    plt.title(title)
    
    # Save with high dpi
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_similarity_matrix(reconstructions, class_averages, save_path, n_colors=None, ood_indices=None):
    """Plot similarity matrix between reconstructions and class averages using SSIM."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
    
    # Initialize ColoredMNIST handler if needed
    mnist = ColoredMNIST(n_colors) if n_colors is not None else None
    
    # Convert to RGB if colored MNIST
    if mnist is not None:
        recon_images = [mnist.to_rgb(r) for r in reconstructions]
        avg_images = [mnist.to_rgb(a) for a in class_averages]
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
    
    # Plot separate matrices for ID and OOD if needed
    if ood_indices is not None:
        # Create masks for ID and OOD samples
        id_mask = np.ones(n_samples, dtype=bool)
        id_mask[ood_indices] = False
        ood_mask = ~id_mask
        
        # Plot ID matrix
        plot_similarity_submatrix(
            similarity_matrix[id_mask][:, id_mask],
            normalized_matrix[id_mask][:, id_mask],
            save_path.replace('.png', '_id.png'),
            n_colors=n_colors,
            indices=np.where(id_mask)[0],
            title_suffix=' - ID Only'
        )
        
        # Plot OOD matrix
        plot_similarity_submatrix(
            similarity_matrix[ood_mask][:, ood_mask],
            normalized_matrix[ood_mask][:, ood_mask],
            save_path.replace('.png', '_ood.png'),
            n_colors=n_colors,
            indices=np.where(ood_mask)[0],
            title_suffix=' - OOD Only'
        )
    else:
        # Plot full matrix
        plot_similarity_submatrix(
            similarity_matrix,
            normalized_matrix,
            save_path,
            n_colors=n_colors,
            indices=np.arange(n_samples)
        )

def plot_similarity_submatrix(matrix, normalized_matrix, save_path, n_colors=None, indices=None, title_suffix=''):
    """Helper function to plot a similarity matrix with proper labels."""
    # Initialize ColoredMNIST handler if needed
    mnist = ColoredMNIST(n_colors) if n_colors is not None else None
    
    for matrix_to_plot, suffix in [(matrix, ''), (normalized_matrix, '_normalized')]:
        fig = plt.figure(figsize=(8, 7))
        ax = plt.gca()
        vmin, vmax = (-1, 1) if suffix == '' else (0, 1)
        im = ax.imshow(matrix_to_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Add colorbar with scientific styling
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('SSIM' + (' (Normalized)' if suffix == '_normalized' else ''), 
                          rotation=270, labelpad=15)
        
        # Configure axes
        ax.set_xticks(np.arange(len(matrix_to_plot)))
        ax.set_yticks(np.arange(len(matrix_to_plot)))
        
        # Generate labels
        labels = []
        for idx in indices:
            digit = idx % 10
            color = idx // 10 if mnist else None
            if mnist:
                is_ood = idx in indices[indices >= len(matrix_to_plot)]
                label = mnist.get_label(digit, color, mark_ood=is_ood)
            else:
                label = str(digit)
            labels.append(label)
        
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        ax.set_xlabel('Class Average')
        ax.set_ylabel('Reconstruction')
        
        # Add similarity values
        for i in range(len(matrix_to_plot)):
            for j in range(len(matrix_to_plot)):
                text = ax.text(j, i, f"{matrix_to_plot[i, j]:.2f}",
                             ha="center", va="center",
                             color="black" if abs(matrix_to_plot[i, j]) < 0.5 else "white",
                             fontsize=8)
        
        plt.title("StR-rec Similarity Matrix" + title_suffix + (' (Normalized)' if suffix == '_normalized' else ''))
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