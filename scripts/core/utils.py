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
import os
import pandas as pd

# Set publication-quality plot parameters for single-column journal figures
plt.rcParams.update({
    'figure.figsize': (3.5, 2.625),  # Standard single-column width (3.5 inches) with 3:4 aspect ratio
    'font.size': 8,                  # Base font size
    'axes.titlesize': 9,             # Title font size
    'axes.labelsize': 8,             # Axis label font size
    'xtick.labelsize': 7,            # X-axis tick label size
    'ytick.labelsize': 7,            # Y-axis tick label size
    'legend.fontsize': 7,            # Legend font size
    'figure.titlesize': 9,           # Overall figure title size
    'figure.dpi': 300,               # Standard publication DPI
    'savefig.dpi': 300,             # Standard publication DPI for saved figures
    'axes.linewidth': 0.5,           # Thinner lines for axes
    'grid.linewidth': 0.5,           # Thinner lines for grid
    'lines.linewidth': 1.0,          # Line width for plots
    'xtick.major.width': 0.5,        # Thinner major ticks
    'ytick.major.width': 0.5,        # Thinner major ticks
    'xtick.minor.width': 0.5,        # Thinner minor ticks
    'ytick.minor.width': 0.5,        # Thinner minor ticks
    'axes.labelpad': 9,              # Smaller label padding
    'axes.titlepad': 7               # Smaller title padding
})

matrix_cell_fontsize = 4

# Define the ordered base colors
BASE_COLORS = np.array([
    [1.0, 0.0, 0.0],  # Red
    [0.0, 0.0, 1.0],  # Blue
    [0.0, 1.0, 0.0],  # Green
    [1.0, 1.0, 0.0],  # Yellow
    [0.0, 1.0, 1.0],  # Cyan
    [1.0, 0.0, 1.0],  # Magenta
    [1.0, 1.1, 1.0],  # White (moved before Orange)
    [1.0, 0.5, 0.0],  # Orange
    [0.5, 0.0, 0.5],  # Purple
    [0.5, 1.0, 0.0],  # Lime
])

COLORS = {i: BASE_COLORS[:i] for i in range(2, 11)}

COLOR_LABELS = ['r', 'b', 'g', 'y', 'c', 'm', 'w', 'o', 'p', 'l']

class ColoredMNIST:
    """Handles colored MNIST data representation and transformations."""
    
    def __init__(self, n_colors, device='cpu', composition_tracker=None):
        """Initialize ColoredMNIST handler.
        
        Args:
            n_colors: Number of colors to use
            device: Device to store data on ('cpu' or 'gpu')
            composition_tracker: Optional CompositionTracker instance
        """
        self.COLOR_LABELS = COLOR_LABELS
        self.COLORS = COLORS
        self.n_colors = n_colors
        self.device = device
        self.colors = self._to_device(self.COLORS[n_colors])
        self.composition_tracker = composition_tracker
    
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
        # Convert input to CPU numpy if needed
        if HAS_CUDA and isinstance(data, cp.ndarray):
            data = data.get()
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        batch_size = data.shape[0]
        
        # Extract intensity and color channels
        intensity = data[:, :784].reshape(batch_size, 28, 28)
        color_channels = [
            data[:, (784 * (i + 1)):(784 * (i + 2))].reshape(batch_size, 28, 28)
            for i in range(self.n_colors)
        ]
        
        # Initialize RGB output
        rgb = np.zeros((batch_size, 28, 28, 3))
        
        # Get colors in numpy
        colors = self.colors.get() if HAS_CUDA and isinstance(self.colors, cp.ndarray) else self.colors
        
        # Combine channels using color mapping
        for i, color_channel in enumerate(color_channels):
            for j in range(3):  # RGB channels
                rgb[:, :, :, j] += color_channel * colors[i, j]
        
        # Scale by intensity
        rgb = rgb * intensity.reshape(batch_size, 28, 28, 1)
        
        # Ensure values are in [0, 1]
        rgb = np.clip(rgb, 0, 1)
        
        return rgb[0] if data.shape[0] == 1 else rgb
    
    def get_label(self, digit, color, mark_ood=False):
        """Get string label for a digit-color combination.
        
        Args:
            digit: Digit label (0-9)
            color: Color index
            mark_ood: Whether to mark out-of-distribution samples
            
        Returns:
            String label (e.g. 'r5' for red 5, 'r5 (OOD)' for OOD red 5)
        """
        # Check if composition is OOD using tracker if available
        if self.composition_tracker is not None:
            idx = self.composition_tracker.get_composition_index(digit, color)
            is_ood = not self.composition_tracker.is_id(idx)
            mark_ood = mark_ood and is_ood
        
        label = f"{self.COLOR_LABELS[color]}{digit}"
        if mark_ood:
            # label = f"{label} (OOD)"
            label = f"{label}" # for cogsci publication...
        return label
    
    def get_row_label(self, color, is_reconstruction=True):
        """Get row label for reconstruction plots.
        
        Args:
            color: Color index
            is_reconstruction: Whether this is a reconstruction row (vs class average)
            
        Returns:
            String label (e.g. 'Reconstructions' or 'Class averages')
        """
        return "Reconstructions" if is_reconstruction else "Class averages"
    
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
    # Use the global random state
    xp = cp if HAS_CUDA and isinstance(data, cp.ndarray) else np
    indices = xp.arange(len(data))
    xp.random.shuffle(indices)  
    return data[indices[:batch_size]], labels[indices[:batch_size]]

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

def plot_similarity_matrix(matrix, save_path, title="Similarity Matrix", n_colors=None, train_indices=None, title_prefix=""):
    """Plot similarity matrix with proper labels and colorbar."""
    # Convert CuPy array to NumPy if needed
    if HAS_CUDA and isinstance(matrix, cp.ndarray):
        matrix = matrix.get()
    
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
    
    # Create normalized version
    normalized_matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        row_min = normalized_matrix[i].min()
        row_max = normalized_matrix[i].max()
        if row_max > row_min:
            normalized_matrix[i] = (normalized_matrix[i] - row_min) / (row_max - row_min)
    
    # Plot both raw and normalized versions
    for matrix_to_plot, suffix in [(matrix, ''), (normalized_matrix, '_normalized')]:
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = (-1, 1) if suffix == '' else (0, 1)
        im = ax.imshow(matrix_to_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Add colorbar with scientific styling
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('Cosine Similarity' + (' (Normalized)' if suffix == '_normalized' else ''), rotation=270)
        cbar.ax.tick_params()
        
        # Generate labels
        labels = []
        n_digits = 10  # MNIST always has 10 digits
        for i in range(matrix_to_plot.shape[0]):
            if mnist is not None:
                color = i // n_digits  # Color is quotient when divided by n_digits
                digit = i % n_digits   # Digit is remainder when divided by n_digits
                is_ood = train_indices is not None and i not in train_indices
                label = mnist.get_label(digit, color, mark_ood=is_ood)
            else:
                label = str(i)
            labels.append(label)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticklabels(labels)
        
        # Add axis labels
        ax.set_xlabel('Reconstruction')
        ax.set_ylabel('Class Average')
        
        # Add title
        if title == "Similarity Matrix":
            title = "StR-rep Similarity Matrix"
            if is_id:
                title += " - ID Only"
            elif is_ood:
                title += " - OOD Only"
        plt.title(title_prefix + title + (' (Normalized)' if suffix == '_normalized' else ''), pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', f'{suffix}.png'), dpi=300, bbox_inches='tight')
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

def plot_reconstructions(reconstructions, class_averages, save_path, n_colors=None, train_indices=None, title_prefix=""):
    """Plot reconstructions and class averages."""
    plt.close('all')
    
    # Convert CuPy arrays to NumPy if needed
    if HAS_CUDA:
        if isinstance(reconstructions, cp.ndarray):
            reconstructions = reconstructions.get()
        if isinstance(class_averages, cp.ndarray):
            class_averages = class_averages.get()
    
    # Determine if this is colored MNIST based on input size
    is_colored = n_colors is not None and reconstructions.shape[1] > 784
    
    if is_colored:
        # Initialize composition tracker and ColoredMNIST handler
        composition_tracker = CompositionTracker(n_colors, train_indices)
        mnist = ColoredMNIST(n_colors, composition_tracker=composition_tracker)
        
        # Create figure with proper dimensions
        n_digits = 10
        fig = plt.figure(figsize=(10, 2.2*n_colors))
        gs = gridspec.GridSpec(2*n_colors, n_digits + 1, hspace=0.3, wspace=0.05)
        
        # Plot reconstructions and class averages for each color
        for color in range(n_colors):
            # Add row labels
            recon_label = plt.subplot(gs[2*color, 0])
            recon_label.text(0.5, 0.5, "Reconst.", 
                           rotation=90, ha='right', va='center',
                           fontsize=11)
            recon_label.axis('off')
            
            avg_label = plt.subplot(gs[2*color + 1, 0])
            avg_label.text(0.5, 0.5, "Class Avg.",
                         rotation=90, ha='right', va='center',
                         fontsize=11)
            avg_label.axis('off')
            
            # Plot each digit in this color row
            for digit in range(n_digits):
                idx = composition_tracker.get_composition_index(digit, color)
                is_ood = not composition_tracker.is_id(idx)
                
                # Plot reconstruction
                ax = plt.subplot(gs[2*color, digit + 1])
                img_rgb = mnist.to_rgb(reconstructions[idx])
                if HAS_CUDA and isinstance(img_rgb, cp.ndarray):
                    img_rgb = img_rgb.get()
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Add label with OOD marker if needed
                label = mnist.get_label(digit, color, mark_ood=is_ood)
                ax.set_title(label, fontsize=9, pad=-0.25)
                
                # Plot class average
                ax = plt.subplot(gs[2*color + 1, digit + 1])
                img_rgb = mnist.to_rgb(class_averages[idx])
                if HAS_CUDA and isinstance(img_rgb, cp.ndarray):
                    img_rgb = img_rgb.get()
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Add label with OOD marker if needed
                label = mnist.get_label(digit, color, mark_ood=is_ood)
                ax.set_title(label, fontsize=9, pad=-0.25)
    else:
        # Regular MNIST plotting
        n_digits = 10
        fig = plt.figure(figsize=(10, 2.2))
        gs = gridspec.GridSpec(2, n_digits + 1, hspace=0.3, wspace=0.05)
        
        # Add row labels
        recon_label = plt.subplot(gs[0, 0])
        recon_label.text(0.5, 0.5, "Reconst.", 
                        rotation=90, ha='right', va='center',
                        fontsize=11)  # Matched with Colored MNIST
        recon_label.axis('off')
        
        avg_label = plt.subplot(gs[1, 0])
        avg_label.text(0.5, 0.5, "Class Avg.",
                      rotation=90, ha='right', va='center',
                      fontsize=11)  # Matched with Colored MNIST
        avg_label.axis('off')
        
        # Plot each digit
        for digit in range(n_digits):
            # Plot reconstruction
            ax = plt.subplot(gs[0, digit + 1])
            img = reconstructions[digit].reshape(28, 28)
            if HAS_CUDA and isinstance(img, cp.ndarray):
                img = img.get()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(str(digit), fontsize=9, pad=-0.75)  # Matched with Colored MNIST
            
            # Plot class average
            ax = plt.subplot(gs[1, digit + 1])
            img = class_averages[digit].reshape(28, 28)
            if HAS_CUDA and isinstance(img, cp.ndarray):
                img = img.get()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(str(digit), fontsize=9, pad=-0.75)  # Matched with Colored MNIST
    
    # Add title outside of grid with adjusted y position for MNIST
    if is_colored:
        plt.suptitle(title_prefix + "StR Reconstructions", fontsize=12, y=0.92)
    else:
        plt.suptitle(title_prefix + "StR Reconstructions", fontsize=12, y=1.05)  # Adjusted for MNIST
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path, title=None, normalize=False, is_color=False, n_colors=None, is_compositional=False, train_indices=None, title_prefix=""):
    """Plot confusion matrix with proper labels and normalization."""
    plt.close()
    
    # Convert to CPU numpy if needed
    if HAS_CUDA and isinstance(conf_matrix, cp.ndarray):
        conf_matrix = conf_matrix.get()
    
    # Initialize composition tracker and ColoredMNIST handler
    composition_tracker = CompositionTracker(n_colors, train_indices) if n_colors is not None else None
    mnist = ColoredMNIST(n_colors, composition_tracker=composition_tracker) if n_colors is not None else None
    
    # Determine matrix type from save path
    is_training = 'train' in save_path
    is_ood = 'ood' in save_path
    is_id = 'id' in save_path
    is_all = 'all' in save_path  # New flag for ID & OOD plots
    
    # For ID/OOD splits, filter matrix to only show relevant combinations
    # Only apply filtering if matrix hasn't been pre-filtered (check size)
    if is_compositional and composition_tracker is not None:
        n_classes = n_colors * 10  # Total number of possible combinations
        
        # Get the appropriate indices based on matrix type
        if is_all:
            indices = list(range(n_classes))  # Use all indices for ID & OOD plots
        elif is_ood:
            indices = [i for i in range(n_classes) if i not in train_indices]
        else:
            indices = [i for i in range(n_classes) if i in train_indices]
            
        if conf_matrix.shape[0] == n_classes:  # Only filter if matrix hasn't been pre-filtered
            # Create mask for rows/columns based on actual matrix size
            mask = np.zeros(n_classes, dtype=bool)
            
            # Map indices to actual matrix positions
            if is_all:
                mask[:] = True  # Use all positions for ID & OOD plots
            elif is_ood:
                for i in range(n_classes):
                    mask[i] = i not in train_indices
            else:
                for i in range(n_classes):
                    mask[i] = i in train_indices
            
            # Filter matrix only if not showing all
            if not is_all:
                conf_matrix = conf_matrix[mask][:, mask]
    else:
        indices = np.arange(conf_matrix.shape[0])
    
    # Create figure
    fig = plt.figure()
    ax = plt.gca()
    
    # Plot matrix
    if normalize:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        conf_matrix = np.where(row_sums > 0, conf_matrix.astype('float') / row_sums, 0)
    
    im = ax.imshow(conf_matrix, cmap='Blues', vmin=0, vmax=1)
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    if normalize:
        cbar.ax.set_ylabel('Classification Rate', rotation=270)
    else:
        cbar.ax.set_ylabel('Count', rotation=270)
    
    # Generate labels based on matrix type
    if is_color:
        labels = [COLOR_LABELS[i] for i in range(n_colors)]
    elif is_compositional and composition_tracker is not None:
        labels = []
        for i in range(conf_matrix.shape[0]):
            actual_idx = indices[i]
            digit, color = composition_tracker.get_digit_color(actual_idx)
            should_mark_ood = is_ood or (is_all and actual_idx not in train_indices)
            label = mnist.get_label(digit, color, mark_ood=should_mark_ood)
            labels.append(label)
    else:
        labels = [str(i) for i in range(conf_matrix.shape[0])]
    
    # Set labels with proper rotation and alignment
    ax.set_xticks(np.arange(len(conf_matrix)))
    ax.set_yticks(np.arange(len(conf_matrix)))
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(labels)
    
    # Add counts as text
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            # If the max value is <= 1, assume it's already normalized
            is_normalized = conf_matrix.max() <= 1
            text = f'{value:.2f}' if is_normalized else str(int(value))
            color = 'white' if value > (0.5 if is_normalized else conf_matrix.max()/2) else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=matrix_cell_fontsize)
    
    # Add axis labels
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    # Set title with proper formatting
    if title is None:
        # Determine if this is MNIST or Colored MNIST from title_prefix
        dataset_type = title_prefix.strip(": ")
        
        # Determine if this is RtS or Decoder from title
        method = "Decoder" if "decoder" in save_path.lower() else "RtS"
        
        # Determine ID/OOD/Test/Train status
        if dataset_type == "MNIST":
            status = "Train" if is_training else "Test"
        else:  # Colored MNIST
            status = "ID" if is_id else "OOD" if is_ood else "ID & OOD"
        
        # Build title
        if dataset_type == "MNIST":
            title = f"{dataset_type}: {method} {status} Confusion Matrix"
        else:  # Colored MNIST
            if is_color:
                title = f"{dataset_type}: {method} Confusion Matrix (Color)"
            elif not is_compositional:
                title = f"{dataset_type}: {method} Confusion Matrix (Digit)"
            else:
                title = f"{dataset_type}: {method} {status} Confusion Matrix"
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_reconstruction_similarity_matrix(reconstructions, class_averages, save_path, n_colors=None, train_indices=None, title_prefix=""):
    """Plot similarity matrix between reconstructions and class averages using SSIM."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
    
    # Initialize composition tracker and ColoredMNIST handler
    composition_tracker = CompositionTracker(n_colors, train_indices) if n_colors is not None else None
    mnist = ColoredMNIST(n_colors, composition_tracker=composition_tracker) if n_colors is not None else None
    
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
    
    # Plot matrices based on ID/OOD split
    if composition_tracker is not None:
        # Plot ID matrix
        id_matrix = similarity_matrix[composition_tracker.id_mask][:, composition_tracker.id_mask]
        id_norm_matrix = normalized_matrix[composition_tracker.id_mask][:, composition_tracker.id_mask]
        plot_similarity_submatrix(
            id_matrix, id_norm_matrix,
            save_path.replace('.png', '_id.png'),
            composition_tracker=composition_tracker,
            mnist=mnist,
            indices=composition_tracker.id_indices,
            title_suffix=' - ID Only',
            title_prefix=title_prefix
        )
        
        # Plot OOD matrix
        ood_matrix = similarity_matrix[composition_tracker.ood_mask][:, composition_tracker.ood_mask]
        ood_norm_matrix = normalized_matrix[composition_tracker.ood_mask][:, composition_tracker.ood_mask]
        plot_similarity_submatrix(
            ood_matrix, ood_norm_matrix,
            save_path.replace('.png', '_ood.png'),
            composition_tracker=composition_tracker,
            mnist=mnist,
            indices=composition_tracker.ood_indices,
            title_suffix=' - OOD Only',
            title_prefix=title_prefix
        )
        
        # Plot full matrix
        plot_similarity_submatrix(
            similarity_matrix, normalized_matrix,
            save_path,
            composition_tracker=composition_tracker,
            mnist=mnist,
            indices=np.arange(n_samples),
            title_prefix=title_prefix
        )
    else:
        # Plot regular matrix for non-compositional case
        plot_similarity_submatrix(
            similarity_matrix, normalized_matrix,
            save_path,
            indices=np.arange(n_samples),
            title_prefix=title_prefix
        )

def plot_similarity_submatrix(matrix, normalized_matrix, save_path, composition_tracker=None, mnist=None, indices=None, title_suffix='', title_prefix=""):
    """Helper function to plot a similarity matrix with proper labels."""
    # Convert CuPy arrays to NumPy if needed
    if HAS_CUDA:
        if isinstance(matrix, cp.ndarray):
            matrix = matrix.get()
        if isinstance(normalized_matrix, cp.ndarray):
            normalized_matrix = normalized_matrix.get()
    
    for matrix_to_plot, suffix in [(matrix, ''), (normalized_matrix, '_normalized')]:
        fig = plt.figure()
        ax = plt.gca()
        vmin, vmax = (-1, 1) if suffix == '' else (0, 1)
        im = ax.imshow(matrix_to_plot, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        
        # Add colorbar with scientific styling
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel('SSIM' + (' (Normalized)' if suffix == '_normalized' else ''), 
                          rotation=270)
        
        # Generate labels
        if composition_tracker is not None and mnist is not None:
            labels = []
            for idx in indices:
                digit, color = composition_tracker.get_digit_color(idx)
                is_ood = not composition_tracker.is_id(idx)
                label = mnist.get_label(digit, color, mark_ood=is_ood)
                labels.append(label)
        else:
            labels = [str(i) for i in range(len(matrix_to_plot))]
        
        # Set labels with proper rotation and alignment
        ax.set_xticks(np.arange(len(matrix_to_plot)))
        ax.set_yticks(np.arange(len(matrix_to_plot)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add similarity values with appropriate size and color
        for i in range(len(matrix_to_plot)):
            for j in range(len(matrix_to_plot)):
                value = matrix_to_plot[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=matrix_cell_fontsize)
        
        # Add axis labels
        ax.set_xlabel('Reconstruction')
        ax.set_ylabel('Class Average')
        
        # Set title with proper formatting
        dataset_type = title_prefix.strip(": ")
        
        if dataset_type == "MNIST":
            title = f"{dataset_type}: StR Similarity Matrix"
        else:  # Colored MNIST
            status = "ID" if "id" in save_path else "OOD" if "ood" in save_path else ""
            matrix_type = ""
            if "digit" in save_path:
                matrix_type = " (Digit)"
            elif "color" in save_path:
                matrix_type = " (Color)"
            
            if status:
                title = f"{dataset_type}: StR {status} Similarity Matrix{matrix_type}"
            else:
                title = f"{dataset_type}: StR Similarity Matrix{matrix_type}"
        
        if suffix == '_normalized':
            title += " (Normalized)"
        
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', f'{suffix}.png'), bbox_inches='tight')
        plt.close()

def compute_compositional_confusion_matrix(digit_preds, color_preds, digit_labels, color_labels):
    """Compute confusion matrix for compositional predictions."""
    # Convert to CPU numpy if needed
    if HAS_CUDA:
        if isinstance(digit_preds, cp.ndarray):
            digit_preds = digit_preds.get()
        if isinstance(color_preds, cp.ndarray):
            color_preds = color_preds.get()
        if isinstance(digit_labels, cp.ndarray):
            digit_labels = digit_labels.get()
        if isinstance(color_labels, cp.ndarray):
            color_labels = color_labels.get()
    
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

def compute_confusion_matrix(true_labels, pred_labels, num_classes=None):
    """Compute confusion matrix."""
    # Convert labels to numpy arrays and ensure integer type
    true_labels = np.asarray(true_labels).astype(np.int64)
    pred_labels = np.asarray(pred_labels).astype(np.int64)
    
    if num_classes is None:
        num_classes = int(max(np.max(true_labels), np.max(pred_labels)) + 1)
    else:
        num_classes = int(num_classes)
    
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1
    return conf_matrix

class CompositionTracker:
    """Tracks ID/OOD compositions and provides consistent labeling."""
    
    def __init__(self, n_colors, train_indices=None):
        """Initialize composition tracker.
        
        Args:
            n_colors: Number of colors in dataset
            train_indices: List of indices that are in-distribution
        """
        self.n_colors = n_colors
        self.n_digits = 10
        self.n_compositions = n_colors * self.n_digits
        
        # Create mapping of all possible compositions
        self.compositions = [(d, c) for d in range(self.n_digits) for c in range(n_colors)]
        
        # Track ID/OOD status
        if train_indices is not None:
            self.train_indices = set(train_indices)  # Convert to set for O(1) lookup
            self.id_mask = np.zeros(self.n_compositions, dtype=bool)
            for idx in train_indices:
                self.id_mask[idx] = True
        else:
            self.train_indices = set(range(self.n_compositions))
            self.id_mask = np.ones(self.n_compositions, dtype=bool)
        
        self.ood_mask = ~self.id_mask
        
        # Get ID and OOD indices
        self.id_indices = np.where(self.id_mask)[0]
        self.ood_indices = np.where(self.ood_mask)[0]
    
    def get_composition_index(self, digit, color):
        """Get flat index for a digit-color composition."""
        return digit * self.n_colors + color
    
    def get_digit_color(self, index):
        """Get digit and color from flat index."""
        digit = index // self.n_colors
        color = index % self.n_colors
        return digit, color
    
    def is_id(self, index):
        """Check if a composition is in-distribution."""
        return index in self.train_indices
    
    def filter_matrix(self, matrix, keep_ood=False):
        """Filter matrix to keep only ID or OOD compositions."""
        mask = self.ood_mask if keep_ood else self.id_mask
        return matrix[mask][:, mask]
    
    def get_filtered_indices(self, keep_ood=False):
        """Get indices for ID or OOD compositions."""
        return self.ood_indices if keep_ood else self.id_indices 

def compute_matrix_metrics(matrix, is_normalized=False):
    """Compute metrics for similarity/reconstruction matrices."""
    # Get diagonal and off-diagonal elements
    diag = np.diag(matrix)
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[off_diag_mask].reshape(matrix.shape[0], -1)
    
    # Compute average diagonal and off-diagonal values
    avg_diag = np.mean(diag)
    avg_off_diag = np.mean(off_diag)
    
    # Compute diagonal vs column average difference for each class
    col_avgs = np.mean(matrix, axis=0)
    diag_vs_col = diag - col_avgs
    
    return {
        f"{'norm_' if is_normalized else ''}avg_diag": avg_diag,
        f"{'norm_' if is_normalized else ''}avg_off_diag": avg_off_diag,
        f"{'norm_' if is_normalized else ''}diag_off_diff": avg_diag - avg_off_diag,
        **{f"{'norm_' if is_normalized else ''}class_{i}_vs_col": val 
           for i, val in enumerate(diag_vs_col)}
    }

def compute_colored_matrix_metrics(matrix, n_colors, train_indices=None):
    """Compute metrics for colored MNIST matrices, split by ID/OOD."""
    metrics = {}
    
    # Create masks for ID/OOD
    n_combinations = matrix.shape[0]
    if train_indices is not None:
        id_mask = np.zeros(n_combinations, dtype=bool)
        id_mask[train_indices] = True
        ood_mask = ~id_mask
        
        # Compute metrics for ID/OOD submatrices
        id_matrix = matrix[id_mask][:, id_mask]
        ood_matrix = matrix[ood_mask][:, ood_mask]
        
        metrics.update({
            f"id_{k}": v for k, v in compute_matrix_metrics(id_matrix).items()
        })
        metrics.update({
            f"ood_{k}": v for k, v in compute_matrix_metrics(ood_matrix).items()
        })
    
    # Compute metrics for digit-only and color-only
    digit_matrix = np.zeros((10, 10))
    color_matrix = np.zeros((n_colors, n_colors))
    
    for i in range(10):
        for j in range(10):
            # Average over all color pairs for each digit pair
            digit_rows = slice(i * n_colors, (i + 1) * n_colors)
            digit_cols = slice(j * n_colors, (j + 1) * n_colors)
            digit_matrix[i, j] = np.mean(matrix[digit_rows, digit_cols])
    
    for i in range(n_colors):
        for j in range(n_colors):
            # Average over all digit pairs for each color pair
            color_values = []
            for d1 in range(10):
                for d2 in range(10):
                    color_values.append(matrix[d1 * n_colors + i, d2 * n_colors + j])
            color_matrix[i, j] = np.mean(color_values)
    
    metrics.update({
        f"digit_{k}": v for k, v in compute_matrix_metrics(digit_matrix).items()
    })
    metrics.update({
        f"color_{k}": v for k, v in compute_matrix_metrics(color_matrix).items()
    })
    
    # Add full matrix metrics
    metrics.update({
        f"all_{k}": v for k, v in compute_matrix_metrics(matrix).items()
    })
    
    return metrics

def save_experiment_metrics(results_dir, net_config, train_config, results):
    """Save detailed experiment metrics to CSV."""
    metrics = {}
    
    # Add all hyperparameters
    metrics.update({
        f"net_{k}": v for k, v in vars(net_config).items()
    })
    metrics.update({
        f"train_{k}": v for k, v in vars(train_config).items()
    })
    
    # Add overall accuracies
    is_colored = 'train_digit_acc_rts' in results
    if is_colored:
        # Colored MNIST metrics
        for split in ['train', 'test']:
            metrics.update({
                f"{split}_digit_acc": results[f'{split}_digit_acc_rts'],
                f"{split}_color_acc": results[f'{split}_color_acc_rts'],
                f"{split}_full_acc": results[f'{split}_full_acc_rts']
            })
    else:
        # Regular MNIST metrics
        metrics.update({
            'train_acc': results['train_acc_rts'],
            'test_acc': results['test_acc_rts']
        })
    
    # Add per-class accuracies from confusion matrices
    if is_colored:
        for split in ['train', 'test']:
            # Normalize confusion matrices
            digit_conf = results[f'{split}_digit_conf_rts']
            digit_conf = digit_conf / digit_conf.sum(axis=1, keepdims=True)
            color_conf = results[f'{split}_color_conf_rts']
            color_conf = color_conf / color_conf.sum(axis=1, keepdims=True)
            
            # Add per-class accuracies
            metrics.update({
                f"{split}_digit_{i}_acc": digit_conf[i, i]
                for i in range(digit_conf.shape[0])
            })
            metrics.update({
                f"{split}_color_{i}_acc": color_conf[i, i]
                for i in range(color_conf.shape[0])
            })
    else:
        for split in ['train', 'test']:
            conf = results[f'{split}_conf_rts']
            conf = conf / conf.sum(axis=1, keepdims=True)
            metrics.update({
                f"{split}_class_{i}_acc": conf[i, i]
                for i in range(conf.shape[0])
            })
    
    # Add StR-rep similarity metrics
    str_rep_matrix = results['str_rep_similarity']
    if is_colored:
        metrics.update(
            compute_colored_matrix_metrics(
                str_rep_matrix, 
                net_config.n_colors, 
                train_indices=train_config.train_indices
            )
        )
    else:
        metrics.update(compute_matrix_metrics(str_rep_matrix))
    
    # Create normalized version of str_rep matrix
    norm_str_rep = str_rep_matrix.copy()
    for i in range(str_rep_matrix.shape[0]):
        row_min = norm_str_rep[i].min()
        row_max = norm_str_rep[i].max()
        if row_max > row_min:
            norm_str_rep[i] = (norm_str_rep[i] - row_min) / (row_max - row_min)
    
    # Add normalized StR-rep metrics
    if is_colored:
        metrics.update(
            compute_colored_matrix_metrics(
                norm_str_rep,
                net_config.n_colors,
                train_indices=train_config.train_indices
            )
        )
    else:
        metrics.update(compute_matrix_metrics(norm_str_rep, is_normalized=True))
    
    # Add StR-rec similarity metrics if available
    if 'str_rec_similarities' in results:
        # Convert similarities list to matrix
        n = len(results['str_rec_similarities'])
        str_rec_matrix = np.zeros((n, n))
        for i in range(n):
            str_rec_matrix[i, i] = results['str_rec_similarities'][i]
        
        if is_colored:
            metrics.update(
                compute_colored_matrix_metrics(
                    str_rec_matrix,
                    net_config.n_colors,
                    train_indices=train_config.train_indices
                )
            )
        else:
            metrics.update(compute_matrix_metrics(str_rec_matrix))
        
        # Create normalized version
        norm_str_rec = str_rec_matrix.copy()
        for i in range(str_rec_matrix.shape[0]):
            row_min = norm_str_rec[i].min()
            row_max = norm_str_rec[i].max()
            if row_max > row_min:
                norm_str_rec[i] = (norm_str_rec[i] - row_min) / (row_max - row_min)
        
        # Add normalized metrics
        if is_colored:
            metrics.update(
                compute_colored_matrix_metrics(
                    norm_str_rec,
                    net_config.n_colors,
                    train_indices=train_config.train_indices
                )
            )
        else:
            metrics.update(compute_matrix_metrics(norm_str_rec, is_normalized=True))
    
    # Save to CSV
    csv_path = os.path.join(results_dir, 'results.csv')
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

def frobenius_norm_from_identity(matrix, return_numpy=False):
    """Compute Frobenius norm difference from identity matrix.
    
    Args:
        matrix: Input matrix
        return_numpy: If True, always return a NumPy array
    """
    # Determine which array module to use based on input type
    xp = cp if isinstance(matrix, cp.ndarray) else np
    identity = xp.eye(matrix.shape[0])
    result = xp.linalg.norm(matrix - identity, ord='fro')
    
    # Convert to NumPy if requested
    if return_numpy and HAS_CUDA and isinstance(result, cp.ndarray):
        result = result.get()
    return result

def set_random_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if HAS_CUDA:
        cp.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass 

def plot_ood_reconstructions(reconstructions, class_averages, save_path, n_colors, train_indices, title_prefix=""):
    """Plot OOD reconstructions and class averages."""
    plt.close('all')
    
    # Convert CuPy arrays to NumPy if needed
    if HAS_CUDA:
        if isinstance(reconstructions, cp.ndarray):
            reconstructions = reconstructions.get()
        if isinstance(class_averages, cp.ndarray):
            class_averages = class_averages.get()
    
    # Initialize composition tracker and ColoredMNIST handler
    composition_tracker = CompositionTracker(n_colors, train_indices)
    mnist = ColoredMNIST(n_colors, composition_tracker=composition_tracker)
    
    # Create figure with proper dimensions for one pair of rows
    n_digits = 10
    fig = plt.figure(figsize=(10, 2.2))
    gs = gridspec.GridSpec(2, n_digits + 1, hspace=0.3, wspace=0.05)
    
    # Add row labels
    recon_label = plt.subplot(gs[0, 0])
    recon_label.text(0.5, 0.5, "Reconst.", 
                    rotation=90, ha='right', va='center',
                    fontsize=11)
    recon_label.axis('off')
    
    avg_label = plt.subplot(gs[1, 0])
    avg_label.text(0.5, 0.5, "Class Avg.",
                  rotation=90, ha='right', va='center',
                  fontsize=11)
    avg_label.axis('off')
    
    # Track plotted digits for each color
    plotted_count = 0
    
    # Plot OOD reconstructions and class averages
    for digit in range(n_digits):
        for color in range(n_colors):
            idx = composition_tracker.get_composition_index(digit, color)
            if not composition_tracker.is_id(idx):  # Only plot if OOD
                # Plot reconstruction
                ax = plt.subplot(gs[0, plotted_count + 1])
                img_rgb = mnist.to_rgb(reconstructions[idx])
                if HAS_CUDA and isinstance(img_rgb, cp.ndarray):
                    img_rgb = img_rgb.get()
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Add label with OOD marker
                label = mnist.get_label(digit, color, mark_ood=True)
                ax.set_title(label, fontsize=9, pad=-0.25)
                
                # Plot class average
                ax = plt.subplot(gs[1, plotted_count + 1])
                img_rgb = mnist.to_rgb(class_averages[idx])
                if HAS_CUDA and isinstance(img_rgb, cp.ndarray):
                    img_rgb = img_rgb.get()
                ax.imshow(img_rgb)
                ax.axis('off')
                
                # Add label with OOD marker
                label = mnist.get_label(digit, color, mark_ood=True)
                ax.set_title(label, fontsize=9, pad=-0.25)
                
                plotted_count += 1
    
    # Add title
    plt.suptitle(title_prefix + "StR OOD Reconstructions", fontsize=12, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 