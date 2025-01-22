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
    load_mnist, get_minibatch, visualize_weights,
    compute_class_averages, plot_association_matrix,
    cosine_similarity, plot_reconstructions, plot_confusion_matrix,
    plot_reconstruction_similarity_matrix
)
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import scipy.io
from datetime import datetime
import argparse

class SimpleDecoder:
    """Linear classifier for evaluation."""
    def __init__(self):
        self.W = None
        
    def fit(self, X, y):
        Y = np.zeros((len(y), 10))
        Y[np.arange(len(y)), y] = 1
        self.W = np.linalg.pinv(X) @ Y
        
    def predict(self, X):
        return np.argmax(X @ self.W, axis=1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

def visualize_weights(net):
    """Visualize weights of first layer as MNIST digits."""
    W = to_device(net.layers[0].W, 'cpu')  # Get first layer weights
    
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
    return plt.gcf()

def compute_confusion_matrix(true_labels, pred_labels, num_classes=10):
    """Compute confusion matrix."""
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        conf_matrix[t, p] += 1
    return conf_matrix

def train_hebbian_mnist(
    layer_sizes=[784, 100],
    signific_size=10,
    hidden_signific_connections=[True],
    batch_size=100,
    eval_batch_size=1000,  # Larger batch size for evaluation
    n_epochs=20,  # Increased from 5
    learning_rate=0.02,
    p=2.0,
    delta=0.4,
    k=2,
    allow_pathway_interaction=False,
    save_dir='results'
):
    """Train network exactly as in KH with our evaluation additions."""
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
    
    # Load data exactly as in KH
    print("Loading MNIST data...")
    M, labels, M_test, labels_test = load_mnist()
    
    # Create signific patterns (1-hot)
    signific_patterns = cp.zeros((len(M), signific_size)) if HAS_CUDA else np.zeros((len(M), signific_size))
    signific_patterns[cp.arange(len(M)) if HAS_CUDA else np.arange(len(M)), labels] = 1.0
    
    print("Initializing network...")
    net = HebbianNetwork(
        layer_sizes=layer_sizes,
        signific_size=signific_size,
        hidden_signific_connections=hidden_signific_connections,
        p=p,
        delta=delta,
        k=k,
        device='gpu' if HAS_CUDA else 'cpu',
        allow_pathway_interaction=allow_pathway_interaction
    )
    
    # Training exactly as in KH
    print("\nTraining network...")
    n_batches = len(M) // batch_size
    learning_rates = np.linspace(learning_rate, 0, n_epochs)  # Linear annealing
    print(f"Running for {n_epochs} epochs with {n_batches} batches per epoch")
    
    for epoch in trange(n_epochs, desc="Epochs"):
        # Shuffle data exactly as in KH
        perm = cp.random.permutation(len(M)) if HAS_CUDA else np.random.permutation(len(M))
        M_epoch = M[perm]
        signific_epoch = signific_patterns[perm]
        
        # Batch training exactly as in KH
        for batch in trange(n_batches, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            batch_data = M_epoch[start_idx:end_idx]
            batch_signific = signific_epoch[start_idx:end_idx]
            net.update(batch_data, signific_pattern=batch_signific, learning_rate=learning_rates[epoch])
        
        # Save weights visualization
        if (epoch + 1) % 5 == 0:
            fig = visualize_weights(net)
            plt.savefig(os.path.join(run_dir, f'weights_epoch_{epoch+1}.png'))
            plt.close(fig)
    
    print("\nEvaluating...")
    # Get representations
    train_repr = []
    for i in trange(0, len(M), eval_batch_size, desc="Computing train representations"):
        batch = M[i:i+eval_batch_size]
        h = to_device(net.forward(batch), 'cpu').T
        train_repr.append(h)
    train_repr = np.vstack(train_repr)
    
    test_repr = []
    for i in trange(0, len(M_test), eval_batch_size, desc="Computing test representations"):
        batch = M_test[i:i+eval_batch_size]
        h = to_device(net.forward(batch), 'cpu').T
        test_repr.append(h)
    test_repr = np.vstack(test_repr)
    
    # Move labels to CPU for evaluation
    labels_cpu = to_device(labels, 'cpu')
    labels_test_cpu = to_device(labels_test, 'cpu')
    
    # Train decoder
    print("Training decoder...")
    decoder = SimpleDecoder()
    decoder.fit(train_repr, labels_cpu)
    
    # Get accuracies
    print("Computing accuracies...")
    train_acc_decoder = decoder.score(train_repr, labels_cpu)
    test_acc_decoder = decoder.score(test_repr, labels_test_cpu)
    
    # Get RtS predictions
    train_preds_rts = []
    for i in trange(0, len(M), eval_batch_size, desc="Computing train RtS predictions"):
        batch = M[i:i+eval_batch_size]
        preds = net.rts_classify(batch, [signific_size])[0]
        train_preds_rts.append(to_device(preds, 'cpu'))
    train_preds_rts = np.concatenate(train_preds_rts)
    train_acc_rts = np.mean(train_preds_rts == labels_cpu)
    
    test_preds_rts = []
    for i in trange(0, len(M_test), eval_batch_size, desc="Computing test RtS predictions"):
        batch = M_test[i:i+eval_batch_size]
        preds = net.rts_classify(batch, [signific_size])[0]
        test_preds_rts.append(to_device(preds, 'cpu'))
    test_preds_rts = np.concatenate(test_preds_rts)
    test_acc_rts = np.mean(test_preds_rts == labels_test_cpu)
    
    # Compute confusion matrices
    train_conf_rts = compute_confusion_matrix(labels_cpu, train_preds_rts)
    test_conf_rts = compute_confusion_matrix(labels_test_cpu, test_preds_rts)
    train_conf_decoder = compute_confusion_matrix(labels_cpu, decoder.predict(train_repr))
    test_conf_decoder = compute_confusion_matrix(labels_test_cpu, decoder.predict(test_repr))
    
    # Save confusion matrices
    plot_confusion_matrix(train_conf_rts, os.path.join(run_dir, 'train_confusion_rts.png'),
                         title='Training Confusion Matrix (RtS)', normalize=True)
    plot_confusion_matrix(test_conf_rts, os.path.join(run_dir, 'test_confusion_rts.png'),
                         title='Test Confusion Matrix (RtS)', normalize=True)
    plot_confusion_matrix(train_conf_decoder, os.path.join(run_dir, 'train_confusion_decoder.png'),
                         title='Training Confusion Matrix (Decoder)', normalize=True)
    plot_confusion_matrix(test_conf_decoder, os.path.join(run_dir, 'test_confusion_decoder.png'),
                         title='Test Confusion Matrix (Decoder)', normalize=True)
    
    # Print and save results
    print(f"\nResults:")
    print("Supervised Decoder:")
    print(f"Train accuracy: {train_acc_decoder:.4f}")
    print(f"Test accuracy: {test_acc_decoder:.4f}")
    print("\nRtS Classification:")
    print(f"Train accuracy: {train_acc_rts:.4f}")
    print(f"Test accuracy: {test_acc_rts:.4f}")
    
    # Compute class averages
    print("\nComputing StR-rep association matrix...")
    class_averages = compute_class_averages(M_test, labels_test, 10, xp=cp if HAS_CUDA else np)
    
    # Create signific inputs for each class
    signific_inputs = cp.eye(10) if HAS_CUDA else np.eye(10)
    
    # Compute association matrix
    association_matrix = cp.zeros((10, 10)) if HAS_CUDA else np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            association_matrix[i, j] = net.str_rep(
                signific_inputs[i:i+1], 
                class_averages[j:j+1]
            )
    
    # Plot association matrix
    plot_association_matrix(
        association_matrix,
        os.path.join(run_dir, 'str_rep_association.png'),
        title="StR-rep Association Matrix"
    )
    
    # Compute StR-rec reconstructions
    print("\nComputing StR-rec reconstructions...")
    reconstructions = []
    reconstruction_similarities = []
    for i in trange(10, desc="Computing reconstructions"):
        # Get reconstruction for each class
        recon = net.str_rec(signific_inputs[i:i+1], max_iter=100)  # Reduced from 1000
        reconstructions.append(recon[0])
        
        # Compare with class average
        sim = cosine_similarity(
            [recon], 
            [class_averages[i:i+1]], 
            xp=cp if HAS_CUDA else np
        )
        reconstruction_similarities.append(float(to_device(sim, 'cpu')))
    
    # Stack reconstructions
    reconstructions = cp.stack(reconstructions) if HAS_CUDA else np.stack(reconstructions)
    
    # Plot reconstructions vs class averages
    plot_reconstructions(
        reconstructions,
        class_averages,
        os.path.join(run_dir, 'str_rec_reconstructions.png')
    )
    
    # Plot reconstruction similarity matrix
    plot_reconstruction_similarity_matrix(
        reconstructions,
        class_averages,
        os.path.join(run_dir, 'str_rec_similarity.png')
    )
    
    # Save summary
    with open(os.path.join(run_dir, 'summary.txt'), 'w') as f:
        f.write("MNIST Experiment Results\n")
        f.write("=======================\n\n")
        
        f.write("Supervised Decoder:\n")
        f.write(f"Train accuracy: {train_acc_decoder:.4f}\n")
        f.write(f"Test accuracy: {test_acc_decoder:.4f}\n\n")
        
        f.write("RtS Classification:\n")
        f.write(f"Train accuracy: {train_acc_rts:.4f}\n")
        f.write(f"Test accuracy: {test_acc_rts:.4f}\n\n")
        
        f.write("StR-rep Cosine Similarities:\n")
        f.write("---------------------------\n")
        
        # Calculate diagonal mean
        diag_mean = float(to_device(
            cp.diag(association_matrix).mean() if HAS_CUDA 
            else np.diag(association_matrix).mean(), 'cpu'
        ))
        f.write(f"Average diagonal (matching classes): {diag_mean:.4f}\n")
        
        # Calculate off-diagonal mean
        off_diag_mean = float(to_device(
            (association_matrix.sum() - cp.diag(association_matrix).sum()) / (100 - 10) if HAS_CUDA 
            else (association_matrix.sum() - np.diag(association_matrix).sum()) / (100 - 10), 'cpu'
        ))
        f.write(f"Average off-diagonal (non-matching classes): {off_diag_mean:.4f}\n")
        
        # Write full matrix
        f.write("\nFull association matrix:\n")
        matrix_cpu = to_device(association_matrix, 'cpu')
        for i in range(10):
            row_str = " ".join(f"{x:.4f}" for x in matrix_cpu[i])
            f.write(row_str + "\n")
            
        f.write("\nStR-rec Reconstruction Similarities:\n")
        f.write("--------------------------------\n")
        f.write(f"Average similarity: {np.mean(reconstruction_similarities):.4f}\n")
        f.write("\nPer-class similarities:\n")
        for i, sim in enumerate(reconstruction_similarities):
            f.write(f"Class {i}: {sim:.4f}\n")
    
    # Save results
    results = {
        'train_acc_decoder': train_acc_decoder,
        'test_acc_decoder': test_acc_decoder,
        'train_acc_rts': train_acc_rts,
        'test_acc_rts': test_acc_rts,
        'train_conf_rts': train_conf_rts,
        'test_conf_rts': test_conf_rts,
        'train_conf_decoder': train_conf_decoder,
        'test_conf_decoder': test_conf_decoder,
        'str_rep_association': to_device(association_matrix, 'cpu'),
        'str_rec_reconstructions': to_device(reconstructions, 'cpu'),
        'str_rec_similarities': reconstruction_similarities
    }
    np.save(os.path.join(run_dir, 'results.npy'), results)
    
    return net, decoder, (train_acc_decoder, test_acc_decoder, train_acc_rts, test_acc_rts), run_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow_pathway_interaction', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--layer_sizes', type=str, help='Comma-separated list of layer sizes')
    parser.add_argument('--hidden_signific_connections', type=str, help='Comma-separated list of boolean values')
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()
    
    # Parse layer sizes from string
    layer_sizes = [int(x.strip()) for x in args.layer_sizes.strip('[]').split(',')]
    
    # Parse signific connections from string
    hidden_signific_connections = [x.strip().lower() == 'true' 
                                 for x in args.hidden_signific_connections.strip('[]').split(',')]
    
    net, decoder, accuracies, run_dir = train_hebbian_mnist(
        layer_sizes=layer_sizes,
        hidden_signific_connections=hidden_signific_connections,
        n_epochs=args.n_epochs,
        allow_pathway_interaction=args.allow_pathway_interaction
    ) 