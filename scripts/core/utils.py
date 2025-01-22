import numpy as np
try:
    import cupy as cp
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    cp = np
import scipy.io
import matplotlib.pyplot as plt

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
    """Visualize weights exactly as in KH's implementation."""
    if HAS_CUDA and isinstance(W, cp.ndarray):
        W = W.get()
    
    # Get dimensions
    n_units = W.shape[0]
    Kx = min(10, int(np.sqrt(n_units)))
    Ky = min(10, int(np.ceil(n_units / Kx)))
    
    # Create heatmap exactly as in KH
    HM = np.zeros((28*Ky, 28*Kx))
    yy = 0
    for y in range(Ky):
        for x in range(Kx):
            if yy < n_units:
                HM[y*28:(y+1)*28, x*28:(x+1)*28] = W[yy].reshape(28, 28)
            yy += 1
    
    # Plot exactly as in KH
    plt.clf()
    nc = np.amax(np.abs(HM))
    plt.imshow(HM, cmap='bwr', vmin=-nc, vmax=nc)
    plt.colorbar(ticks=[np.amin(HM), 0, np.amax(HM)])
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
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
    """Plot association matrix between signific and referential inputs."""
    if HAS_CUDA and isinstance(matrix, cp.ndarray):
        matrix = matrix.get()
        
    fig, ax = plt.figure(figsize=(10, 10)), plt.gca()
    im = ax.imshow(matrix, cmap='Blues', vmin=-1, vmax=1)
    
    plt.colorbar(im)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xlabel('Referential Class')
    ax.set_ylabel('Signific Class')
    
    # Add numbers
    for i in range(10):
        for j in range(10):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                         ha="center", va="center", 
                         color="black" if matrix[i, j] < 0.5 else "white")
    
    plt.title(title)
    plt.savefig(save_path)
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
    """Plot reconstructed images alongside class averages."""
    if HAS_CUDA:
        reconstructions = reconstructions.get()
        class_averages = class_averages.get()
        
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    
    # Plot reconstructions
    for i in range(10):
        axes[0, i].imshow(reconstructions[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Recon {i}')
        
        axes[1, i].imshow(class_averages[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Avg {i}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 