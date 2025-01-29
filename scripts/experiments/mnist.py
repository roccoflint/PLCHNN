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
    compute_class_averages, plot_similarity_matrix,
    cosine_similarity, plot_reconstructions, plot_confusion_matrix,
    plot_reconstruction_similarity_matrix, save_experiment_metrics,
    frobenius_norm_from_identity, compute_gmcc
)
from core.config import NetworkConfig, TrainingConfig
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import scipy.io
from datetime import datetime
import argparse
from core.stats import test_accuracy, test_gmcc, test_frobenius, format_metric

def format_gmcc(value, n_samples):
    """Format GMCC with significance stars."""
    if value is None:
        return "N/A"
    p_value = test_gmcc(value, n_samples)
    return format_metric(value, p_value)

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

def train_hebbian_mnist(net_config, train_config):
    """Train network exactly as in KH's implementation."""
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
    
    # Load data exactly as in KH
    print("Loading MNIST data...")
    M, labels, M_test, labels_test = load_mnist()
    
    # Create signific patterns (1-hot)
    signific_patterns = cp.zeros((len(M), net_config.signific_size)) if HAS_CUDA else np.zeros((len(M), net_config.signific_size))
    signific_patterns[cp.arange(len(M)) if HAS_CUDA else np.arange(len(M)), labels] = 1.0
    
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
    
    # Training exactly as in KH
    print("\nTraining network...")
    n_batches = len(M) // train_config.batch_size
    learning_rates = np.linspace(train_config.learning_rate, 0, train_config.n_epochs)  # Linear annealing
    print(f"Running for {train_config.n_epochs} epochs with {n_batches} batches per epoch")
    
    for epoch in trange(train_config.n_epochs, desc="Epochs"):
        # Shuffle data exactly as in KH
        perm = cp.random.permutation(len(M)) if HAS_CUDA else np.random.permutation(len(M))
        M_epoch = M[perm]
        signific_epoch = signific_patterns[perm]
        
        # Batch training exactly as in KH
        for batch in trange(n_batches, desc=f"Epoch {epoch+1}/{train_config.n_epochs}", leave=False):
            start_idx = batch * train_config.batch_size
            end_idx = start_idx + train_config.batch_size
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
    for i in trange(0, len(M), train_config.eval_batch_size, desc="Computing train representations"):
        batch = M[i:i+train_config.eval_batch_size]
        h = to_device(net.forward(batch), 'cpu').T
        train_repr.append(h)
    train_repr = np.vstack(train_repr)
    
    test_repr = []
    for i in trange(0, len(M_test), train_config.eval_batch_size, desc="Computing test representations"):
        batch = M_test[i:i+train_config.eval_batch_size]
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
    for i in trange(0, len(M), train_config.eval_batch_size, desc="Computing train RtS predictions"):
        batch = M[i:i+train_config.eval_batch_size]
        preds = net.rts_classify(batch, [net_config.signific_size])[0]
        train_preds_rts.append(to_device(preds, 'cpu'))
    train_preds_rts = np.concatenate(train_preds_rts)
    train_acc_rts = np.mean(train_preds_rts == labels_cpu)
    
    test_preds_rts = []
    for i in trange(0, len(M_test), train_config.eval_batch_size, desc="Computing test RtS predictions"):
        batch = M_test[i:i+train_config.eval_batch_size]
        preds = net.rts_classify(batch, [net_config.signific_size])[0]
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
    train_acc_decoder_p = test_accuracy(train_acc_decoder, 10, len(M))
    test_acc_decoder_p = test_accuracy(test_acc_decoder, 10, len(M_test))
    print(f"Train accuracy: {format_metric(train_acc_decoder, train_acc_decoder_p)}")
    print(f"Test accuracy: {format_metric(test_acc_decoder, test_acc_decoder_p)}")
    
    print("\nRtS Classification:")
    train_acc_rts_p = test_accuracy(train_acc_rts, 10, len(M))
    test_acc_rts_p = test_accuracy(test_acc_rts, 10, len(M_test))
    print(f"Train accuracy: {format_metric(train_acc_rts, train_acc_rts_p)}")
    print(f"Test accuracy: {format_metric(test_acc_rts, test_acc_rts_p)}")
    
    # Compute class averages
    print("\nComputing StR-rep similarity matrix...")
    class_averages = compute_class_averages(M_test, labels_test, 10, xp=cp if HAS_CUDA else np)
    
    # Create signific inputs for each class
    signific_inputs = cp.eye(10) if HAS_CUDA else np.eye(10)
    
    # Compute similarity matrix
    similarity_matrix = cp.zeros((10, 10)) if HAS_CUDA else np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            similarity_matrix[i, j] = net.str_rep(
                signific_inputs[i:i+1], 
                class_averages[j:j+1].reshape(1, -1)  # Add reshape to match colored MNIST
            )
    
    # Plot similarity matrix
    plot_similarity_matrix(
        similarity_matrix,
        os.path.join(run_dir, 'str_rep_similarity.png'),
        title="StR-rep Similarity Matrix"
    )
    
    # Compute StR-rec reconstructions
    print("\nComputing StR-rec reconstructions...")
    reconstructions = []
    reconstruction_similarities = []
    for i in trange(10, desc="Computing reconstructions"):
        # Get reconstruction for each class
        recon = net.str_rec(signific_inputs[i:i+1])
        reconstructions.append(recon[0])
        
        # Compare with class average
        sim = cosine_similarity(
            [recon],
            [class_averages[i:i+1].reshape(1, -1)],  # Add reshape to match colored MNIST
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
        f.write("======================\n\n")
        
        # StR-rep metrics
        f.write("StR-rep Metrics:\n")
        f.write("---------------\n")
        str_rep_fn = frobenius_norm_from_identity(similarity_matrix, return_numpy=True)
        str_rep_fn_p = test_frobenius(str_rep_fn, 10)
        
        # Create normalized version
        norm_str_rep = similarity_matrix.copy()
        for i in range(norm_str_rep.shape[0]):
            row_min = norm_str_rep[i].min()
            row_max = norm_str_rep[i].max()
            if row_max > row_min:
                norm_str_rep[i] = (norm_str_rep[i] - row_min) / (row_max - row_min)
        
        str_rep_fn_norm = frobenius_norm_from_identity(norm_str_rep, return_numpy=True)
        str_rep_fn_norm_p = test_frobenius(str_rep_fn_norm, 10)
        
        f.write(f"StR-rep FN: {format_metric(str_rep_fn, str_rep_fn_p)}\n")
        f.write(f"StR-rep FN Normalized: {format_metric(str_rep_fn_norm, str_rep_fn_norm_p)}\n\n")
        
        # StR-rec metrics
        f.write("StR-rec Metrics:\n")
        f.write("---------------\n")
        str_rec_matrix = np.zeros((10, 10))
        # Compute full similarity matrix for reconstructions
        for i in range(10):
            for j in range(10):
                sim = cosine_similarity(
                    [to_device(reconstructions[i:i+1], 'cpu')],
                    [to_device(class_averages[j:j+1].reshape(1, -1), 'cpu')],  # Add reshape to match colored MNIST
                    xp=np
                )
                str_rec_matrix[i, j] = float(sim)
        
        str_rec_fn = frobenius_norm_from_identity(str_rec_matrix, return_numpy=True)
        str_rec_fn_p = test_frobenius(str_rec_fn, 10)
        
        # Create normalized version
        norm_str_rec = str_rec_matrix.copy()
        for i in range(norm_str_rec.shape[0]):
            row_min = norm_str_rec[i].min()
            row_max = norm_str_rec[i].max()
            if row_max > row_min:
                norm_str_rec[i] = (norm_str_rec[i] - row_min) / (row_max - row_min)
        
        str_rec_fn_norm = frobenius_norm_from_identity(norm_str_rec, return_numpy=True)
        str_rec_fn_norm_p = test_frobenius(str_rec_fn_norm, 10)
        
        f.write(f"StR-rec FN: {format_metric(str_rec_fn, str_rec_fn_p)}\n")
        f.write(f"StR-rec FN Normalized: {format_metric(str_rec_fn_norm, str_rec_fn_norm_p)}\n\n")
        
        # Classification metrics
        f.write("Classification Metrics:\n")
        f.write("---------------------\n")
        
        # Compute GMCCs
        train_gmcc_rts = compute_gmcc(train_conf_rts)
        test_gmcc_rts = compute_gmcc(test_conf_rts)
        train_gmcc_decoder = compute_gmcc(train_conf_decoder)
        test_gmcc_decoder = compute_gmcc(test_conf_decoder)
        
        f.write(f"RtS Train GMCC: {format_gmcc(train_gmcc_rts, len(M))}\n")
        f.write(f"RtS Test GMCC: {format_gmcc(test_gmcc_rts, len(M_test))}\n")
        f.write(f"Decoder Train GMCC: {format_gmcc(train_gmcc_decoder, len(M))}\n")
        f.write(f"Decoder Test GMCC: {format_gmcc(test_gmcc_decoder, len(M_test))}\n")
        f.write(f"RtS Train Accuracy: {format_metric(train_acc_rts, train_acc_rts_p)}\n")
        f.write(f"RtS Test Accuracy: {format_metric(test_acc_rts, test_acc_rts_p)}\n")
        f.write(f"Decoder Train Accuracy: {format_metric(train_acc_decoder, train_acc_decoder_p)}\n")
        f.write(f"Decoder Test Accuracy: {format_metric(test_acc_decoder, test_acc_decoder_p)}\n\n")
        
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
        
        f.write("RtS Confusion Matrix (Train):\n")
        for i in range(train_conf_rts.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in train_conf_rts[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("RtS Confusion Matrix (Test):\n")
        for i in range(test_conf_rts.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in test_conf_rts[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("Decoder Confusion Matrix (Train):\n")
        for i in range(train_conf_decoder.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in train_conf_decoder[i])
            f.write(f"{row_str}\n")
        f.write("\n")
        
        f.write("Decoder Confusion Matrix (Test):\n")
        for i in range(test_conf_decoder.shape[0]):
            row_str = " ".join(f"{x:.4f}" for x in test_conf_decoder[i])
            f.write(f"{row_str}\n")
        f.write("\n")
    
    # Save results
    results = {
        'train_acc_decoder': train_acc_decoder,
        'test_acc_decoder': test_acc_decoder,
        'train_acc_rts': train_acc_rts,
        'test_acc_rts': test_acc_rts,
        'train_conf_decoder': train_conf_decoder,
        'test_conf_decoder': test_conf_decoder,
        'train_conf_rts': train_conf_rts,
        'test_conf_rts': test_conf_rts,
        'str_rep_similarity': to_device(similarity_matrix, 'cpu'),
        'str_rec_reconstructions': to_device(reconstructions, 'cpu'),
        'str_rec_similarities': reconstruction_similarities,
        'str_rec_matrix': to_device(str_rec_matrix, 'cpu')  # Add full StR-rec similarity matrix
    }
    np.save(os.path.join(run_dir, 'results.npy'), results)
    
    # Save detailed metrics
    save_experiment_metrics(run_dir, net_config, train_config, results)
    
    return net, results, run_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--p', type=float, default=3.0)
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--delta', type=float, default=0.4)
    parser.add_argument('--signific_p_multiplier', type=float, default=4.0)
    parser.add_argument('--allow_pathway_interaction', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.02)
    args = parser.parse_args()
    
    net_config = NetworkConfig(
        hidden_sizes=args.hidden_size,
        p=args.p,
        k=args.k,
        delta=args.delta,
        signific_p_multiplier=args.signific_p_multiplier,
        allow_pathway_interaction=args.allow_pathway_interaction
    )
    
    train_config = TrainingConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    net, results, run_dir = train_hebbian_mnist(net_config, train_config) 