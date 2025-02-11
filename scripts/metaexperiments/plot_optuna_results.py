import optuna
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Get absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Load the studies
colored_mnist_study = optuna.load_study(
    storage=f"sqlite:///{os.path.join(project_root, 'results/optuna_colored_mnist/study.db')}",
    study_name=None  # Will load the existing study name
)

mnist_study = optuna.load_study(
    storage=f"sqlite:///{os.path.join(project_root, 'results/optuna_mnist/study.db')}",
    study_name=None  # Will load the existing study name
)

# Create results directory for plots
plots_dir = os.path.join(project_root, "results/optuna_plots")
os.makedirs(plots_dir, exist_ok=True)

# Helper function to plot optimization history
def plot_optimization_history(study, target_idx=None, dataset_name=""):
    trials = study.trials
    if target_idx is not None:
        values = [t.values[target_idx] for t in trials if t.values is not None]
    else:
        values = [t.value for t in trials if t.value is not None]
    
    plt.figure(figsize=(10, 6))
    plt.plot(values, marker='o')
    plt.xlabel('Trial')
    plt.ylabel('Objective Value')
    plt.title(f'{dataset_name} Optimization History')
    plt.grid(True)

# Helper function to plot parameter importances
def plot_param_importances(study, target_idx=None, dataset_name=""):
    """Plot parameter importances using Optuna's FANOVA-based importance evaluator.
    
    The importance values are normalized to sum to 1.0 and represent each parameter's
    contribution to the variance in the objective value. Higher values indicate
    parameters that have a stronger effect on the optimization outcome.
    """
    if target_idx is not None:
        importances = optuna.importance.get_param_importances(study, target=lambda t: t.values[target_idx])
    else:
        importances = optuna.importance.get_param_importances(study)
    
    # Verify importances sum to approximately 1
    importance_sum = sum(importances.values())
    assert abs(importance_sum - 1.0) < 1e-6, f"Importance values should sum to 1, got {importance_sum}"
    
    plt.figure(figsize=(10, 6))
    params = list(importances.keys())
    scores = list(importances.values())
    
    # Sort by importance
    sorted_indices = np.argsort(scores)
    params = [params[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    plt.barh(range(len(params)), scores)
    plt.yticks(range(len(params)), params)
    
    # Add numerical values on bars
    for i, score in enumerate(scores):
        plt.text(score, i, f' {score:.3f}', va='center')
    
    plt.xlabel('Importance Score (FANOVA-based contribution to objective variance)')
    plt.title(f'{dataset_name} Hyperparameter Importance')
    plt.tight_layout()

# Create a summary file with all plottable data
def write_summary(study, dataset_name, target_idx=None, summary_file=None):
    if summary_file is None:
        summary_file = os.path.join(project_root, f"results/optuna_plots/{dataset_name.lower().replace(' ', '_')}_summary.txt")
    
    # Append if target_idx is not None (for multi-objective), otherwise create new file
    mode = "a" if target_idx is not None else "w"
    with open(summary_file, mode) as f:
        if target_idx is None:
            f.write(f"{dataset_name} Optimization Summary\n")
            f.write("="*50 + "\n\n")
            f.write("Note on Parameter Importance Values:\n")
            f.write("The importance values are calculated using Functional ANOVA (FANOVA) and represent\n")
            f.write("each parameter's contribution to the variance in the objective value.\n")
            f.write("Values are normalized to sum to 1.0. Higher values indicate parameters that\n")
            f.write("have a stronger effect on the optimization outcome.\n\n")
        
        if target_idx is not None:
            f.write(f"\nObjective {target_idx}:\n")
            f.write("-"*50 + "\n")
        
        # Get optimization history
        trials = study.trials
        if target_idx is not None:
            values = [t.values[target_idx] for t in trials if t.values is not None]
            f.write(f"Optimization History:\n")
        else:
            values = [t.value for t in trials if t.value is not None]
            f.write("Optimization History:\n")
        
        for trial_idx, value in enumerate(values):
            f.write(f"Trial {trial_idx}: {value}\n")
        
        # Get parameter importances
        if target_idx is not None:
            importances = optuna.importance.get_param_importances(study, target=lambda t: t.values[target_idx])
            f.write(f"\nParameter Importances:\n")
        else:
            importances = optuna.importance.get_param_importances(study)
            f.write("\nParameter Importances:\n")
        
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{param}: {importance:.4f}\n")
        
        f.write("\n")
        return importances

def calculate_average_importances(importances_list):
    avg_importances = defaultdict(float)
    param_counts = defaultdict(int)
    
    # Sum up importances
    for importances in importances_list:
        for param, importance in importances.items():
            avg_importances[param] += importance
            param_counts[param] += 1
    
    # Calculate averages
    for param in avg_importances:
        avg_importances[param] /= param_counts[param]
    
    return dict(avg_importances)

# Plot for colored MNIST (multi-objective)
colored_mnist_importances = []
for i in range(len(colored_mnist_study.directions)):
    # Optimization history
    plot_optimization_history(colored_mnist_study, target_idx=i, dataset_name=f"Colored MNIST (Objective {i})")
    plt.savefig(os.path.join(plots_dir, f"colored_mnist_optimization_history_obj{i}.png"))
    plt.close()
    
    # Parameter importance
    plot_param_importances(colored_mnist_study, target_idx=i, dataset_name=f"Colored MNIST (Objective {i})")
    plt.savefig(os.path.join(plots_dir, f"colored_mnist_param_importances_obj{i}.png"))
    plt.close()
    
    # Write summary and collect importances
    importances = write_summary(colored_mnist_study, "Colored MNIST", target_idx=i)
    colored_mnist_importances.append(importances)

# Calculate and write average importances for Colored MNIST
avg_importances = calculate_average_importances(colored_mnist_importances)
with open(os.path.join(project_root, "results/optuna_plots/colored_mnist_summary.txt"), "a") as f:
    f.write("\nAverage Parameter Importances Across All Objectives:\n")
    f.write("-"*50 + "\n")
    for param, importance in sorted(avg_importances.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{param}: {importance:.4f}\n")

# Plot average parameter importances for Colored MNIST
plt.figure(figsize=(10, 6))
params = list(avg_importances.keys())
scores = list(avg_importances.values())

# Sort by importance
sorted_indices = np.argsort(scores)
params = [params[i] for i in sorted_indices]
scores = [scores[i] for i in sorted_indices]

plt.barh(range(len(params)), scores)
plt.yticks(range(len(params)), params)

# Add numerical values on bars
for i, score in enumerate(scores):
    plt.text(score, i, f' {score:.3f}', va='center')

plt.xlabel('Average Importance Score')
plt.title('Colored MNIST Average Parameter Importance')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "colored_mnist_average_param_importances.png"))
plt.close()

# Plot for MNIST (single-objective)
# Optimization history
plot_optimization_history(mnist_study, dataset_name="MNIST")
plt.savefig(os.path.join(plots_dir, "mnist_optimization_history.png"))
plt.close()

# Parameter importance
plot_param_importances(mnist_study, dataset_name="MNIST")
plt.savefig(os.path.join(plots_dir, "mnist_param_importances.png"))
plt.close()

# Write summary
write_summary(mnist_study, "MNIST")

print(f"Plots and summary have been saved to {plots_dir}/")
print("Generated files:")
print("- colored_mnist_optimization_history_obj[0-N].png")
print("- mnist_optimization_history.png")
print("- colored_mnist_param_importances_obj[0-N].png")
print("- colored_mnist_average_param_importances.png")
print("- mnist_param_importances.png")
print("- colored_mnist_summary.txt")
print("- mnist_summary.txt") 