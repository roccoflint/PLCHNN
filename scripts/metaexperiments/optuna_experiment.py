import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import torch
import json
from sklearn.metrics import matthews_corrcoef
from datetime import datetime
import matplotlib.pyplot as plt
from core.config import NetworkConfig, TrainingConfig
from experiments.mnist import train_hebbian_mnist
from experiments.colored_mnist import train_hebbian_colored_mnist
from core.stats import test_gmcc

def frobenius_norm_from_identity(matrix):
    """Compute Frobenius norm difference from identity matrix."""
    identity = np.eye(matrix.shape[0])
    return np.linalg.norm(matrix - identity, ord='fro')

def compute_gmcc(conf_matrix):
    """Compute Generalized Matthews Correlation Coefficient from confusion matrix."""
    n_classes = conf_matrix.shape[0]
    
    # Convert to probabilities
    conf_matrix = conf_matrix / conf_matrix.sum()
    
    # Compute GMCC components
    cov_x_y = 0
    cov_x_x = 0
    cov_y_y = 0
    
    row_sums = conf_matrix.sum(axis=1)
    col_sums = conf_matrix.sum(axis=0)
    
    for i in range(n_classes):
        for j in range(n_classes):
            cov_x_y += (i * j) * conf_matrix[i, j]
            cov_x_x += (i * i) * row_sums[i]
            cov_y_y += (j * j) * col_sums[j]
    
    mean_x = sum(i * row_sums[i] for i in range(n_classes))
    mean_y = sum(j * col_sums[j] for j in range(n_classes))
    
    cov_x_y -= mean_x * mean_y
    cov_x_x -= mean_x * mean_x
    cov_y_y -= mean_y * mean_y
    
    if cov_x_x * cov_y_y == 0:
        return 0
    
    return cov_x_y / np.sqrt(cov_x_x * cov_y_y)

def compute_mnist_loss(results_dict):
    """Compute loss for MNIST experiment."""
    # Extract matrices
    str_rec_matrix = results_dict['str_rec_matrix']
    test_conf_rts = results_dict['test_conf_rts']
    
    # Compute raw FN from StR-rec matrix (already using cosine similarities)
    str_rec_fn = frobenius_norm_from_identity(str_rec_matrix)
    
    # Compute GMCC from test confusion matrix
    rts_gmcc = compute_gmcc(test_conf_rts)
    
    # Handle failed GMCC computation
    if rts_gmcc is None or np.isnan(rts_gmcc):
        rts_gmcc = -1  # Worst possible GMCC
    
    # Equal weighting between StR-rec and RtS performance
    # Note: str_rec_fn should be minimized, while rts_gmcc should be maximized
    # rts_gmcc is in [-1, 1], so we use (1 - rts_gmcc) / 2 to get [0, 1]
    loss = 0.5 * str_rec_fn + 0.5 * (1 - rts_gmcc)
    
    return loss

def compute_colored_mnist_loss(results_dict, train_indices):
    """Compute loss for colored MNIST experiment, focusing only on OOD performance."""
    try:
        # Extract matrices
        str_rec_matrix = results_dict['str_rec_matrix']
        test_comp_conf_rts = results_dict['test_comp_conf_rts']
        
        # Create OOD mask
        n_combinations = str_rec_matrix.shape[0]
        id_mask = np.zeros(n_combinations, dtype=bool)
        id_mask[train_indices] = True
        ood_mask = ~id_mask
        
        # Get OOD matrices
        str_rec_ood = str_rec_matrix[ood_mask][:, ood_mask]
        test_conf_ood = test_comp_conf_rts[ood_mask][:, ood_mask]
        
        # Compute OOD StR-rec Frobenius norm
        ood_fn = frobenius_norm_from_identity(str_rec_ood)
        
        # Compute OOD GMCC
        ood_gmcc = compute_gmcc(test_conf_ood)
        
        # Handle failed GMCC computation
        if ood_gmcc is None or np.isnan(ood_gmcc):
            ood_gmcc = -1  # Worst possible GMCC
        
        # Handle NaN in Frobenius norm
        if np.isnan(ood_fn):
            ood_fn = float('inf')  # Worst possible FN
        
        # Equal weighting between OOD StR-rec and OOD RtS performance
        # Note: ood_fn should be minimized, while ood_gmcc should be maximized
        # ood_gmcc is in [-1, 1], so we use (1 - ood_gmcc) / 2 to get [0, 1]
        loss = 0.5 * ood_fn + 0.5 * (1 - ood_gmcc)
        
        # Final safety check
        if np.isnan(loss) or np.isinf(loss):
            return float('inf')  # Return worst possible loss
            
        return loss
        
    except Exception as e:
        print(f"Error in loss computation: {str(e)}")
        return float('inf')

def objective_mnist(trial):
    """Objective function for MNIST optimization."""
    # Sample hyperparameters
    net_config = NetworkConfig(
        hidden_sizes=trial.suggest_int('hidden_size', 100, 324),
        p=trial.suggest_float('p', 2.0, 4.0),
        k=trial.suggest_int('k', 1, 7),
        delta=trial.suggest_float('delta', 0.2, 0.6),
        signific_p_multiplier=trial.suggest_float('signific_p_multiplier', 1.0, 4.0),
        allow_pathway_interaction=trial.suggest_categorical('allow_pathway_interaction', [True, False])
    )
    
    train_config = TrainingConfig(
        n_epochs=trial.suggest_int('n_epochs', 10, 60),
        batch_size=trial.suggest_int('batch_size', 64, 128),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.03, log=True)
    )
    
    # Train network
    net, results, run_dir = train_hebbian_mnist(net_config, train_config)
    
    # Load results
    results_dict = np.load(os.path.join(run_dir, 'results.npy'), allow_pickle=True).item()
    
    # Compute single combined loss
    loss = compute_mnist_loss(results_dict)
    
    return loss

def objective_colored_mnist(trial):
    """Objective function for colored MNIST optimization."""
    # Sample hyperparameters
    net_config = NetworkConfig(
        hidden_sizes=trial.suggest_int('hidden_size', 100, 324),
        n_colors=2,  # Fixed for this experiment
        p=trial.suggest_float('p', 2.0, 4.0),
        k=trial.suggest_int('k', 1, 7),
        delta=trial.suggest_float('delta', 0.2, 0.6),
        signific_p_multiplier=trial.suggest_float('signific_p_multiplier', 1.0, 4.0),
        allow_pathway_interaction=trial.suggest_categorical('allow_pathway_interaction', [True, False])
    )
    
    train_config = TrainingConfig(
        n_epochs=trial.suggest_int('n_epochs', 10, 60),
        batch_size=trial.suggest_int('batch_size', 64, 128),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.03, log=True),
        train_ratio=0.8  # Fixed for this experiment
    )
    
    # Train network
    net, results_dict, run_dir = train_hebbian_colored_mnist(net_config, train_config)
    
    # Get train indices from results
    train_indices = results_dict['train_indices']
    
    # Compute single combined loss
    loss = compute_colored_mnist_loss(results_dict, train_indices)
    
    return loss

def run_optuna_experiment(experiment_type, n_trials=100, study_name=None):
    """Run Optuna experiment for either MNIST or colored MNIST."""
    if study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f'optuna_{experiment_type}_optimization_{timestamp}'
    
    # Create results directory
    results_dir = os.path.join('results', study_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create study with single objective
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # Single objective to minimize
        storage=f'sqlite:///{results_dir}/study.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True  # Enable constant liar to improve parallel optimization
        )
    )
    
    # Create callback to track best value
    best_values = []
    def track_best_value(study, trial):
        history = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if history:
            best_values.append(min(history))
        else:
            best_values.append(float('inf'))
        
        # Plot optimization history in real-time
        plt.figure(figsize=(10, 6))
        plt.plot(best_values, 'b-', label='Best Value')
        plt.xlabel('Trial')
        plt.ylabel('Loss Value')
        plt.title(f'Optimization Progress - {experiment_type.upper()}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'optimization_progress.png'))
        plt.close()
    
    # Run optimization with callback
    print(f"\nStarting optimization for {experiment_type}...")
    objective = objective_mnist if experiment_type == 'mnist' else objective_colored_mnist
    study.optimize(objective, n_trials=n_trials, callbacks=[track_best_value])
    
    # Save study statistics
    study_stats = {
        'pareto_front': [
            {
                'number': t.number,
                'values': [float(v) for v in t.values],  # Convert numpy types to native Python
                'params': t.params
            }
            for t in study.best_trials
        ],
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'best_value_history': best_values  # Add best value history
    }
    
    # Print optimization summary
    print("\nOptimization Summary:")
    print(f"Number of completed trials: {study_stats['n_complete']}")
    print(f"Number of pruned trials: {study_stats['n_pruned']}")
    if best_values:
        print(f"Best value achieved: {best_values[-1]:.4f}")
    
    # Save study stats to JSON
    with open(os.path.join(results_dir, 'study_stats.json'), 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    # Generate standard Optuna visualizations
    try:
        # Optimization history plot
        history_plot = optuna.visualization.plot_optimization_history(study)
        history_plot.write_image(os.path.join(results_dir, 'optuna_history.png'))
        
        # Parameter importance plot
        importance_plot = optuna.visualization.plot_param_importances(study)
        importance_plot.write_image(os.path.join(results_dir, 'param_importances.png'))
        
        # Parallel coordinate plot
        parallel_plot = optuna.visualization.plot_parallel_coordinate(study)
        parallel_plot.write_image(os.path.join(results_dir, 'parallel_coordinate.png'))
        
        # Slice plot
        slice_plot = optuna.visualization.plot_slice(study)
        slice_plot.write_image(os.path.join(results_dir, 'slice_plot.png'))
        
    except Exception as e:
        print(f"Warning: Could not generate some visualizations: {str(e)}")
    
    return study, study_stats, results_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, choices=['mnist', 'colored_mnist'], required=True)
    parser.add_argument('--n_trials', type=int, default=2)
    parser.add_argument('--continue_from', type=str, help='Directory of previous study to continue from')
    args = parser.parse_args()
    
    if args.continue_from:
        # Extract timestamp from the previous study directory
        study_name = os.path.basename(args.continue_from)
        print(f"\nContinuing optimization from {study_name}...")
        study, stats, results_dir = run_optuna_experiment(args.experiment, args.n_trials, study_name=study_name)
    else:
        study, stats, results_dir = run_optuna_experiment(args.experiment, args.n_trials)
    
    print(f"\nOptimization completed. Results saved in {results_dir}")
    print("\nBest trial:")
    trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
    print(f"Trial {trial['number']}:")
    print(f"  Value: {trial['values'][0]:.4f}")
    print("  Parameters:")
    for param, value in trial['params'].items():
        print(f"    {param}: {value}") 