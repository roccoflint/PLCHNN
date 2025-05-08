import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_ood_confusion_matrix(confusion_str, top_k=5, bar_width=0.2, 
                              plot_title="Colored Mnist: Top 5 predicted classes", 
                              save_path="colored_mnist_top5.png"):
    plt.style.use("default")  # Use standard, light matplotlib style
    
    # Parse the confusion matrix string into a numpy array.
    rows = confusion_str.strip().splitlines()
    matrix = np.array([[float(x) for x in row.split()] for row in rows])
    
    # Define the 50 class labels (digits 0–9, each with colors 'r', 'b', 'g', 'y', 'c')
    base_colors = ['r', 'b', 'g', 'y', 'c']
    class_labels = [f"{color}{digit}" for digit in range(10) for color in base_colors]

    # OOD true classes and their corresponding row indices.
    ood_labels = ['c0', 'y1', 'g2', 'b3', 'r4', 'c5', 'y6', 'g7', 'b8', 'r9']
    ood_indices = [class_labels.index(label) for label in ood_labels]
    
    # Create subplots with tighter spacing.
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), gridspec_kw={"wspace": 0.3, "hspace": 0.7})
    axes = axes.flatten()
    
    for i, (ax, ood_idx, true_label) in enumerate(zip(axes, ood_indices, ood_labels)):
        row = matrix[ood_idx]
        total = row.sum()
        row_norm = row / total if total > 0 else row  # Normalize counts for that true class
        
        # Get top-k predicted classes.
        top_indices = np.argsort(row_norm)[-top_k:][::-1]
        top_counts = row_norm[top_indices]
        top_pred_labels = [class_labels[j] for j in top_indices]
        
        # Color the bars using the Blues colormap, where a value of 1.0 maps to the darkest blue.
        norm = plt.Normalize(0, 1)
        bar_colors = cm.Blues(norm(row_norm[top_indices]))
        
        positions = np.arange(top_k)
        ax.bar(positions, row_norm[top_indices], width=bar_width, color=bar_colors)
        
        # Bold the predicted class label if it exactly matches the true class.
        formatted_labels = [
            r"$\mathbf{" + label + "}$" if label == true_label else label 
            for label in top_pred_labels
        ]
        ax.set_xticks(positions)
        ax.set_xticklabels(formatted_labels, rotation=45, ha='right', fontsize=10)
        ax.set_title(f"True Class: {true_label}", fontsize=12)
        
        # Set y-axis limits from 0 to 1
        ax.set_ylim(0, 1)
        
        # Only the leftmost subplot in each row gets the y-axis label.
        if i % 5 == 0:
            ax.set_ylabel("Normalized Count", fontsize=11)
        ax.set_xlabel("Predicted Class", fontsize=10)
    
    fig.suptitle(plot_title, fontsize=15)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_ood_confusion_matrix(confusion_str, top_k=5, bar_width=0.4, plot_title="Colored Mnist: Top 5 Predicted Classes per OOD True Class")