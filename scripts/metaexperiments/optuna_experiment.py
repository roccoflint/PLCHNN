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
from core.utils import compute_gmcc

def frobenius_norm_from_identity(matrix):
    """Compute Frobenius norm difference from identity matrix."""
    identity = np.eye(matrix.shape[0])
    return np.linalg.norm(matrix - identity, ord='fro')

def compute_mnist_loss(results_dict, use_frobenius=True):
    """Compute loss for MNIST experiment using either GMCC and Frobenius norm or raw accuracies."""
    # Extract matrices
    str_rec_matrix = results_dict['str_rec_matrix']
    test_conf_rts = results_dict['test_conf_rts']
    
    if use_frobenius:
        # Compute Frobenius norms from identity for both matrices
        str_rec_fn = frobenius_norm_from_identity(str_rec_matrix)
        rts_fn = frobenius_norm_from_identity(test_conf_rts / np.sum(test_conf_rts, axis=1, keepdims=True))  # Normalize confusion matrix
        
        # Equal weighting between StR-rec and RtS performance
        loss = 0.5 * str_rec_fn + 0.5 * rts_fn
    else:
        # Use raw accuracies
        str_sim = np.mean(np.diag(str_rec_matrix))  # Average diagonal elements for reconstruction accuracy
        test_acc = np.sum(np.diag(test_conf_rts)) / np.sum(test_conf_rts)  # Classification accuracy
        
        # Convert to losses (1 - accuracy)
        str_loss = 1 - str_sim
        rts_loss = 1 - test_acc
        
        # Equal weighting between StR-rec and RtS
        loss = 0.5 * str_loss + 0.5 * rts_loss
    
    return loss

def compute_colored_mnist_loss(results_dict, train_indices, use_frobenius=True):
    """Compute loss for colored MNIST experiment using either Frobenius norm or raw accuracies."""
    try:
        # Extract matrices and raw accuracies
        str_rec_matrix = results_dict['str_rec_matrix']
        unconstrained_test_comp_conf = results_dict['unconstrained_test_comp_conf']
        
        # Create ID/OOD masks
        n_classes = str_rec_matrix.shape[0]
        id_mask = np.zeros(n_classes, dtype=bool)
        id_mask[train_indices] = True
        ood_mask = ~id_mask
        
        # Split StR matrix into ID and OOD
        str_rec_id = str_rec_matrix[id_mask][:, id_mask]
        str_rec_ood = str_rec_matrix[ood_mask][:, ood_mask]
        
        # Split confusion matrix into ID and OOD
        conf_id = unconstrained_test_comp_conf[id_mask][:, id_mask]
        conf_ood = unconstrained_test_comp_conf[ood_mask][:, ood_mask]
        
        if use_frobenius:
            # Compute Frobenius norm from identity for both ID and OOD
            str_loss_id = frobenius_norm_from_identity(str_rec_id)
            str_loss_ood = frobenius_norm_from_identity(str_rec_ood)
            
            # Normalize confusion matrices and compute Frobenius norms
            conf_id_norm = conf_id / (np.sum(conf_id, axis=1, keepdims=True) + 1e-10)
            conf_ood_norm = conf_ood / (np.sum(conf_ood, axis=1, keepdims=True) + 1e-10)
            rts_loss_id = frobenius_norm_from_identity(conf_id_norm)
            rts_loss_ood = frobenius_norm_from_identity(conf_ood_norm)
            
            # Apply exponential penalty to OOD losses
            str_loss_ood = np.exp(str_loss_ood) - 1
            rts_loss_ood = np.exp(rts_loss_ood) - 1
        else:
            # Use raw accuracies/similarities
            str_sim_id = np.mean(np.diag(str_rec_id))
            str_sim_ood = np.mean(np.diag(str_rec_ood))
            
            # Convert to losses
            str_loss_id = 1 - str_sim_id  # Linear penalty for ID
            str_loss_ood = np.exp(2 * (1 - str_sim_ood)) - 1  # Exponential penalty for OOD
            
            # Compute raw accuracies from confusion matrices
            rts_acc_id = np.sum(np.diag(conf_id)) / (np.sum(conf_id) + 1e-10)
            rts_acc_ood = np.sum(np.diag(conf_ood)) / (np.sum(conf_ood) + 1e-10)
            rts_loss_id = 1 - rts_acc_id  # Linear penalty for ID
            rts_loss_ood = np.exp(2 * (1 - rts_acc_ood)) - 1  # Exponential penalty for OOD
        
        # Handle failed computations
        if np.isnan(str_loss_id) or np.isnan(str_loss_ood) or np.isnan(rts_loss_id) or np.isnan(rts_loss_ood):
            return 10.0
        
        # Weight ID/OOD (with exponential OOD penalty for both cases)
        str_rec_loss = 0.25 * str_loss_id + 0.75 * str_loss_ood
        rts_loss = 0.25 * rts_loss_id + 0.75 * rts_loss_ood
        
        # Equal weighting between StR-rec and RtS
        loss = 0.5 * str_rec_loss + 0.5 * rts_loss
        
        return loss
        
    except Exception as e:
        print(f"Error in loss computation: {str(e)}")
        return 10.0  # Large but finite penalty

def objective_mnist(trial, use_frobenius=True):
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
    loss = compute_mnist_loss(results_dict, use_frobenius)
    
    return loss

def objective_colored_mnist(trial, use_frobenius=True):
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
    loss = compute_colored_mnist_loss(results_dict, train_indices, use_frobenius)
    
    return loss

def run_optuna_experiment(experiment_type, n_trials=100, study_name=None, use_frobenius=True, initial_params=None):
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
    
    # If initial parameters are provided and this is a new study, create a trial with those parameters
    if initial_params is not None and len(study.trials) == 0:
        study.enqueue_trial(initial_params)
    
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
    
    # Create partial function with use_frobenius
    from functools import partial
    objective_with_loss = partial(objective, use_frobenius=use_frobenius)
    
    study.optimize(objective_with_loss, n_trials=n_trials, callbacks=[track_best_value])
    
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
        'best_value_history': best_values,  # Add best value history
        'use_frobenius': use_frobenius  # Save loss type used
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
    parser.add_argument('--continue_from', type=str, help='Path to previous study.db to continue from')
    parser.add_argument('--use_frobenius', action='store_true', help='Use Frobenius norm based loss instead of raw accuracies')
    parser.add_argument('--study_name', type=str, help='Specific name for the study (overrides automatic timestamp)')
    parser.add_argument('--initial_params_file', type=str, help='JSON file containing initial parameters for the study')
    args = parser.parse_args()
    
    # Load initial parameters if provided
    initial_params = None
    if args.initial_params_file:
        with open(args.initial_params_file, 'r') as f:
            initial_params = json.load(f)
    
    if args.continue_from:
        # Extract study name from the previous study directory
        study_name = os.path.basename(os.path.dirname(args.continue_from))
        print(f"\nContinuing optimization from {study_name}...")
        study, stats, results_dir = run_optuna_experiment(args.experiment, args.n_trials, study_name=study_name, use_frobenius=args.use_frobenius)
    else:
        study, stats, results_dir = run_optuna_experiment(args.experiment, args.n_trials, study_name=args.study_name, use_frobenius=args.use_frobenius, initial_params=initial_params)
    
    print(f"\nOptimization completed. Results saved in {results_dir}")
    print("\nBest trial:")
    trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
    print(f"Trial {trial['number']}:")
    print(f"  Value: {trial['values'][0]:.4f}")
    print("  Parameters:")
    for param, value in trial['params'].items():
        print(f"    {param}: {value}") 