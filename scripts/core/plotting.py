import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import optuna
from typing import Optional

# MNIST plotting functions

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

# Colored MNIST specific plotting functions

def plot_ood_confusion_matrix(confusion_str: str,
                              top_k: int = 5,
                              bar_width: float = 0.2,
                              plot_title: str = "Colored MNIST: Top Predicted Classes per OOD True Class",
                              save_path: Optional[str] = None) -> None:
    """
    Parse a confusion matrix string and plot the top-k predicted class probabilities
    for a set of predefined out-of-distribution (OOD) true classes.
    
    The confusion_str should contain rows of numbers (separated by whitespace)
    that form the confusion matrix. The function defines a set of 50 class labels (digits 0–9 in
    5 colors) and a fixed set of OOD true labels.
    
    Parameters:
        confusion_str: String containing the confusion matrix.
        top_k: The number of top predictions to display.
        bar_width: The width of the bars in the output plot.
        plot_title: The title of the figure.
        save_path: If provided, the figure is saved to the specified path.
    """
    # Parse the confusion matrix.
    rows = confusion_str.strip().splitlines()
    matrix = np.array([[float(x) for x in row.split()] for row in rows])
    
    # Define the 50 class labels (each digit 0-9 with 5 colors).
    base_colors = ['r', 'b', 'g', 'y', 'c']
    class_labels = [f"{color}{digit}" for digit in range(10) for color in base_colors]
    
    # Predefined OOD labels (these should correspond to specific rows in the matrix).
    ood_labels = ['c0', 'y1', 'g2', 'b3', 'r4', 'c5', 'y6', 'g7', 'b8', 'r9']
    ood_indices = [class_labels.index(label) for label in ood_labels]
    
    # Create a 2x5 grid of subplots.
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.7})
    axes = axes.flatten()
    
    for ax, ood_idx, true_label in zip(axes, ood_indices, ood_labels):
        row = matrix[ood_idx]
        total = row.sum()
        row_norm = row / total if total > 0 else row
        
        # Identify top-k predictions.
        top_indices = np.argsort(row_norm)[-top_k:][::-1]
        top_pred_labels = [class_labels[j] for j in top_indices]
        
        # Use a Blue colormap for the bars.
        norm = plt.Normalize(0, 1)
        bar_colors = cm.Blues(norm(row_norm[top_indices]))
        
        positions = np.arange(top_k)
        ax.bar(positions, row_norm[top_indices], width=bar_width, color=bar_colors)
        
        # Bold the label if it exactly matches the true class.
        formatted_labels = [f"$\\mathbf{{{label}}}$" if label == true_label else label for label in top_pred_labels]
        ax.set_xticks(positions)
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=10)
        ax.set_title(f"True Class: {true_label}", fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Class", fontsize=10)
        # Only label the y-axis on the leftmost plots in each row.
        if ax in [axes[0], axes[5]]:
            ax.set_ylabel("Normalized Count", fontsize=11)
    
    fig.suptitle(plot_title, fontsize=15)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

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

# Optuna plotting functions

def plot_optimization_history(study, target_idx: Optional[int] = None, dataset_name: str = "", save_path: Optional[str] = None) -> None:
    """
    Plot the optimization history for a given Optuna study.
    
    Parameters:
        study: An Optuna study object.
        target_idx: If provided, the index of the objective for multi-objective studies.
        dataset_name: A name to tag the plot (e.g., 'MNIST') in the title.
        save_path: If provided, save the figure to this path.
    """
    trials = study.trials
    if target_idx is not None:
        values = [t.values[target_idx] for t in trials if t.values is not None]
    else:
        values = [t.value for t in trials if t.value is not None]
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, marker='o')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title(f"{dataset_name} Optimization History")
    ax.grid(True)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_param_importances(study, target_idx: Optional[int] = None, dataset_name: str = "", save_path: Optional[str] = None) -> None:
    """
    Generate a horizontal bar plot of parameter importances for an Optuna study.
    
    Parameters:
        study: An Optuna study object.
        target_idx: For multi-objective studies, the index of the objective to use.
        dataset_name: The name of the dataset/experiment (for titling the plot).
        save_path: If provided, save the figure to this path.
    """
    if target_idx is not None:
        importance = optuna.importance.get_param_importances(study, target=lambda t: t.values[target_idx])
    else:
        importance = optuna.importance.get_param_importances(study)
    
    # Sort parameters by importance.
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    params = [item[0] for item in sorted_importance]
    scores = [item[1] for item in sorted_importance]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(params))
    ax.barh(positions, scores, color='skyblue')
    ax.set_yticks(positions)
    ax.set_yticklabels(params)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"{dataset_name} Parameter Importances")
    for i, score in enumerate(scores):
        ax.text(score, i, f" {score:.3f}", va='center')
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show() 
