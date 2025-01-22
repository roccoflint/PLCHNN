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

def visualize_weights(W, save_path=None):
    """Visualize learned receptive fields."""
    if HAS_CUDA and isinstance(W, cp.ndarray):
        W = W.get()
    
    # Get dimensions
    n_units = W.shape[0]
    Kx = min(10, int(np.sqrt(n_units)))
    Ky = min(10, int(np.ceil(n_units / Kx)))
    
    # Create receptive field grid
    HM = np.zeros((28*Ky, 28*Kx))
    yy = 0
    for y in range(Ky):
        for x in range(Kx):
            if yy < n_units:
                HM[y*28:(y+1)*28, x*28:(x+1)*28] = W[yy].reshape(28, 28)
            yy += 1
    
    # Create figure with specific size for publication
    plt.figure(figsize=(8, 8))
    nc = np.amax(np.abs(HM))
    plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    cbar = plt.colorbar(ticks=[np.amin(HM), 0, np.amax(HM)])
    cbar.ax.set_ylabel('Weight Magnitude', rotation=270, labelpad=15)
    plt.title('Learned Receptive Fields')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def cosine_similarity(v1, v2, xp=np):
    """Compute cosine similarity between two lists of activation vectors."""
    similarities = []
    for a1, a2 in zip(v1, v2):
        norm1 = xp.linalg.norm(a1)
        norm2 = xp.linalg.norm(a2)
        similarities.append(xp.sum(a1 * a2) / (norm1 * norm2))
    return xp.mean(xp.array(similarities))

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

def plot_reconstructions(reconstructions, class_averages, save_path):
    """Plot reconstructed MNIST digits from signific inputs."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
        
    # Create figure and axes grid
    fig, axes = plt.subplots(2, 10, figsize=(14, 5))
    
    # Add overall title
    plt.suptitle('StR Reconstructions per Class', y=1.0)
    
    # Add row titles with larger font
    fig.text(0.5, 0.9, 'Reconstruction', ha='center', va='center', fontsize=14)
    fig.text(0.5, 0.52, 'Class Average', ha='center', va='center', fontsize=14)
    
    # Plot reconstructions and class averages
    for i in range(10):
        # Reconstruction
        axes[0, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(str(i))
        
        # Class average
        axes[1, i].imshow(class_averages[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        axes[1, i].set_title(str(i))
    
    # Adjust spacing - slightly increased hspace
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.1, hspace=0.2)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(conf_matrix, save_path, title=None, normalize=False):
    """Plot confusion matrix with publication-quality styling."""
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    
    # Use a perceptually uniform colormap
    im = ax.imshow(conf_matrix, cmap='Blues', aspect='equal')
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Normalized Count' if normalize else 'Count', 
                      rotation=270, labelpad=15)
    
    # Configure axes
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    
    # Add counts as text
    thresh = conf_matrix.max() / 2
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, format(conf_matrix[i, j], fmt),
                         ha="center", va="center",
                         color="white" if conf_matrix[i, j] > thresh else "black")
    
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_reconstruction_similarity_matrix(reconstructions, class_averages, save_path):
    """Plot similarity matrix between reconstructions and class averages using SSIM."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
    
    # Compute similarity matrix using SSIM
    similarity_matrix = np.zeros((10, 10))
    for i in range(10):
        recon_i = reconstructions[i].reshape(28, 28)
        for j in range(10):
            avg_j = class_averages[j].reshape(28, 28)
            similarity_matrix[i, j] = ssim(recon_i, avg_j, data_range=1.0)
    
    # Plot matrix
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar with scientific styling
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('SSIM', rotation=270, labelpad=15)
    
    # Configure axes
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Class Average')
    ax.set_ylabel('Reconstruction')
    
    # Add similarity values
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                         ha="center", va="center", 
                         color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white",
                         fontsize=8)
    
    plt.title('StR Reconstruction Similarity')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 