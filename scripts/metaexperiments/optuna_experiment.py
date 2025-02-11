import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import torch
import json
from functools import partial
from datetime import datetime
from core.config import NetworkConfig, TrainingConfig
from experiments.mnist import train_hebbian_mnist
from experiments.colored_mnist import train_hebbian_colored_mnist
from core.utils import set_random_seed, frobenius_norm_from_identity

set_random_seed()

def compute_mnist_loss(results_dict, use_frobenius_str=False, use_frobenius_rts=False):
    """Compute loss for MNIST experiment using either GMCC and Frobenius norm or raw accuracies."""
    # Extract matrices
    str_rec_matrix = results_dict['str_rec_matrix']
    test_conf_rts = results_dict['test_conf_rts']
    
    # Calculate raw accuracies/similarities first
    str_sim = np.mean(np.diag(str_rec_matrix))
    test_acc = np.sum(np.diag(test_conf_rts)) / np.sum(test_conf_rts)
    
    print(f"\nAccuracies/Similarities:")
    print(f"StR Diagonal Mean: {str_sim:.3f}")
    print(f"RtS Accuracy: {test_acc:.3f}")
    
    # -- StR LOSS COMPUTATION ------------------------------------------------
    # Normalize Frobenius norm by matrix size (10 classes)
    str_fn = frobenius_norm_from_identity(str_rec_matrix) / 10
    
    # Log-based loss for StR raw similarities
    str_loss = -np.log(str_sim + 1e-10)
    
    # -- RtS LOSS COMPUTATION ------------------------------------------------
    # Normalize confusion matrix row-wise
    conf_norm = test_conf_rts / np.sum(test_conf_rts, axis=1, keepdims=True)
    
    # Normalize Frobenius norm by matrix size
    rts_fn = frobenius_norm_from_identity(conf_norm) / 10
    
    # Log-based loss for RtS Frobenius norm
    rts_loss = -np.log(1 - rts_fn + 1e-10)
    
    # Print intermediate values for debugging
    print(f"\nLosses:")
    print(f"StR Loss (log of raw sim): {str_loss:.4f}")
    print(f"RtS Loss (log of 1-FN): {rts_loss:.4f}")
    
    # Return combined loss
    combined_loss = str_loss + rts_loss
    print(f"Combined Loss: {combined_loss:.4f}")
    
    return combined_loss

def compute_colored_mnist_loss(results_dict, train_indices, use_frobenius_str=False, use_frobenius_rts=False):
    """Compute separate loss components for colored MNIST experiment."""
    try:
        # Extract matrices and raw accuracies
        str_rec_matrix = results_dict['str_rec_matrix']
        unconstrained_test_comp_conf = results_dict['unconstrained_test_comp_conf']
        
        if 'test_comp_conf_rts' not in results_dict:
            print("\nError: Missing test_comp_conf_rts!")
            return (10.0, 10.0, 10.0, 10.0)
        
        constrained_conf = results_dict['test_comp_conf_rts']
        
        # Create OOD mask for evaluations
        n_classes = str_rec_matrix.shape[0]
        id_mask = np.zeros(n_classes, dtype=bool)
        id_mask[train_indices] = True
        ood_mask = ~id_mask
        n_ood = np.sum(ood_mask)
        
        # Get OOD portions
        str_rec_ood = str_rec_matrix[ood_mask][:, ood_mask]
        conf_ood = unconstrained_test_comp_conf[ood_mask][:, ood_mask]
        
        # Calculate accuracies/similarities
        str_sim_all = np.mean(np.diag(str_rec_matrix))
        str_sim_ood = np.mean(np.diag(str_rec_ood))
        rts_acc_all = np.sum(np.diag(unconstrained_test_comp_conf)) / (np.sum(unconstrained_test_comp_conf) + 1e-10)
        
        # Fix: Calculate OOD accuracy considering all possible classes
        ood_rows = unconstrained_test_comp_conf[ood_mask]
        correct_ood = np.sum(np.diag(unconstrained_test_comp_conf)[ood_mask])
        rts_acc_ood = correct_ood / (np.sum(ood_rows) + 1e-10)
        
        print(f"\nAccuracies/Similarities:")
        print(f"StR All: {str_sim_all:.3f}, OOD portion: {str_sim_ood:.3f}")
        print(f"RtS All: {rts_acc_all:.3f}, OOD portion: {rts_acc_ood:.3f}")
        
        # -- StR LOSS COMPUTATION ------------------------------------------------
        if use_frobenius_str:
            # Log-based Frobenius loss
            str_fn_all = frobenius_norm_from_identity(str_rec_matrix) / n_classes
            str_fn_ood = frobenius_norm_from_identity(str_rec_ood) / max(n_ood, 1)
            str_loss_all = -np.log(1 - str_fn_all + 1e-10)
            str_loss_ood = -np.log(1 - str_fn_ood + 1e-10)
        else:
            # Log-based loss for reconstruction
            str_loss_all = -np.log(str_sim_all + 1e-10)
            str_loss_ood = -np.log(str_sim_ood + 1e-10)
            
        # -- RtS LOSS COMPUTATION ------------------------------------------------
        if use_frobenius_rts:
            # RtS Frobenius loss (unconstrained test conf normalized row-wise, then from identity)
            conf_all_norm = unconstrained_test_comp_conf / (
                np.sum(unconstrained_test_comp_conf, axis=1, keepdims=True) + 1e-10
            )
            conf_ood_norm = conf_ood / (
                np.sum(conf_ood, axis=1, keepdims=True) + 1e-10
            )
            rts_fn_all = frobenius_norm_from_identity(conf_all_norm) / n_classes
            rts_fn_ood = frobenius_norm_from_identity(conf_ood_norm) / max(n_ood, 1)
            rts_loss_all = -np.log(1 - rts_fn_all + 1e-10)
            rts_loss_ood = -np.log(1 - rts_fn_ood + 1e-10)
        else:
            # Log-based loss for classification
            rts_loss_all = -np.log(rts_acc_all + 1e-10)
            rts_loss_ood = -np.log(rts_acc_ood + 1e-10)
        
        # Check for NaNs
        if any(np.isnan(x) for x in [str_loss_all, str_loss_ood, rts_loss_all, rts_loss_ood]):
            return (10.0, 10.0, 10.0, 10.0)
        
        return (str_loss_all, str_loss_ood, rts_loss_all, rts_loss_ood)
        
    except Exception as e:
        print(f"Error in loss computation: {str(e)}")
        return (10.0, 10.0, 10.0, 10.0)

def objective_mnist(trial, use_frobenius_str=False, use_frobenius_rts=False):
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
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.045, log=True)
    )
    
    # Train network
    net, results, run_dir = train_hebbian_mnist(net_config, train_config)
    
    # Load results
    results_dict = np.load(os.path.join(run_dir, 'results.npy'), allow_pickle=True).item()
    
    # Compute combined loss using the common function
    combined_loss = compute_mnist_loss(results_dict, use_frobenius_str=True, use_frobenius_rts=True)
    
    # Print current values for debugging
    print(f"\nMNIST Trial Metrics:")
    print(f"Combined Loss: {combined_loss:.4f}")
    
    return combined_loss

def objective_colored_mnist(trial, use_frobenius_str=False, use_frobenius_rts=False):
    """Multi-objective function for colored MNIST optimization."""
    # Sample hyperparameters
    net_config = NetworkConfig(
        hidden_sizes=trial.suggest_int('hidden_size', 100, 324),
        n_colors=5,  # Fixed for this experiment
        p=trial.suggest_float('p', 2.0, 4.0),
        k=trial.suggest_int('k', 1, 7),
        delta=trial.suggest_float('delta', 0.2, 0.6),
        signific_p_multiplier=trial.suggest_float('signific_p_multiplier', 1.0, 4.0),
        allow_pathway_interaction=trial.suggest_categorical('allow_pathway_interaction', [True, False])
    )
    
    train_config = TrainingConfig(
        n_epochs=trial.suggest_int('n_epochs', 10, 60),
        batch_size=trial.suggest_int('batch_size', 64, 128),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.045, log=True),
        train_ratio=0.8  # Fixed for this experiment
    )
    
    # Train network
    net, results_dict, run_dir = train_hebbian_colored_mnist(net_config, train_config)
    
    # Get train indices from results
    train_indices = results_dict['train_indices']
    
    # Compute all loss components
    str_loss_all, str_loss_ood, rts_loss_all, rts_loss_ood = compute_colored_mnist_loss(
        results_dict, train_indices, use_frobenius_str, use_frobenius_rts
    )
    
    return [str_loss_all, str_loss_ood, rts_loss_all, rts_loss_ood]

def run_optuna_experiment(experiment_type, n_trials=100, study_name=None, use_frobenius_str=False, use_frobenius_rts=False, initial_params=None, continue_from=None, seed=42):
    set_random_seed(seed)
    
    # Set up study storage and directory
    if continue_from:
        results_dir = os.path.dirname(continue_from)
        storage_url = f"sqlite:///{continue_from}"
        existing_study = optuna.load_study(
            storage=storage_url,
            study_name=None
        )
        study_name = existing_study.study_name
        print(f"\nContinuing existing study '{study_name}' from {continue_from}")
    else:
        if study_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            study_name = f'optuna_{experiment_type}_optimization_{timestamp}'
        
        results_dir = os.path.join('results', study_name)
        os.makedirs(results_dir, exist_ok=True)
        storage_url = f'sqlite:///{os.path.join(results_dir, "study.db")}'
        print(f"\nCreating new study '{study_name}' in {results_dir}")

    # Create or load study with appropriate configuration based on experiment type
    if experiment_type == 'mnist':
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=seed)
        )
        objective = partial(objective_mnist, use_frobenius_str=use_frobenius_str, use_frobenius_rts=use_frobenius_rts)
    else:  # colored_mnist
        study = optuna.create_study(
            study_name=study_name,
            directions=['minimize'] * 4,  # Four objectives for colored MNIST
            storage=storage_url,
            load_if_exists=True,
            sampler=optuna.samplers.NSGAIISampler(seed=seed)
        )
        objective = partial(objective_colored_mnist, use_frobenius_str=use_frobenius_str, use_frobenius_rts=use_frobenius_rts)
    
    # Print study status
    print(f"\nStudy status:")
    print(f"  Name: {study.study_name}")
    print(f"  Storage: {storage_url}")
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Existing trials: {len(study.trials)}")
    print(f"  Completed trials: {len(completed_trials)}")
    if completed_trials:
        print("\nBest trial so far:")
        if experiment_type == 'mnist':
            best_trial = study.best_trial
            print(f"  Trial {best_trial.number}:")
            print(f"    Combined Loss: {best_trial.value:.4f}")
        else:
            print("\nPareto front:")
            for trial in study.best_trials:
                print(f"  Trial {trial.number}:")
                print(f"    StR All: {trial.values[0]:.4f}")
                print(f"    StR OOD: {trial.values[1]:.4f}")
                print(f"    RtS All: {trial.values[2]:.4f}")
                print(f"    RtS OOD: {trial.values[3]:.4f}")
    
    # If initial parameters are provided and this is a new study, create a trial with those parameters
    if initial_params is not None and len(study.trials) == 0:
        study.enqueue_trial(initial_params)
    
    # Create callback to track best values
    def track_best_values(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"\nTrial {trial.number} completed:")
            if experiment_type == 'mnist':
                print(f"  Combined Loss: {trial.value:.4f}")
                if study.best_trial.number == trial.number:
                    print("  This is the best trial so far!")
            else:
                print(f"  StR All: {trial.values[0]:.4f}")
                print(f"  StR OOD: {trial.values[1]:.4f}")
                print(f"  RtS All: {trial.values[2]:.4f}")
                print(f"  RtS OOD: {trial.values[3]:.4f}")
    
    # Run optimization with callback
    print(f"\nStarting optimization for {experiment_type}...")
    study.optimize(objective, n_trials=n_trials, callbacks=[track_best_values])
    
    # Save study statistics
    study_stats = {
        'best_trial': {
            'number': study.best_trial.number,
            'value': float(study.best_trial.value),
            'params': study.best_trial.params
        } if experiment_type == 'mnist' else None,
        'pareto_front': None if experiment_type == 'mnist' else [
            {
                'number': t.number,
                'values': [float(v) for v in t.values],
                'params': t.params
            }
            for t in study.best_trials
        ],
        'n_trials': len(study.trials),
        'n_complete': len(completed_trials),
        'use_frobenius_str': use_frobenius_str,
        'use_frobenius_rts': use_frobenius_rts,
        'study_name': study.study_name,
        'storage_path': os.path.join(results_dir, "study.db")
    }
    
    # Save study stats to JSON
    with open(os.path.join(results_dir, 'study_stats.json'), 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    return study, study_stats, results_dir

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, choices=['mnist', 'colored_mnist'], required=True)
    parser.add_argument('--n_trials', type=int, default=2)
    parser.add_argument('--continue_from', type=str, help='Path to previous study.db to continue from')
    parser.add_argument('--use_frobenius_str', action='store_true', help='Use Frobenius norm for reconstruction')
    parser.add_argument('--use_frobenius_rts', action='store_true', help='Use Frobenius norm for classification')
    parser.add_argument('--study_name', type=str, help='Specific name for the study (overrides automatic timestamp)')
    parser.add_argument('--initial_params_file', type=str, help='JSON file containing initial parameters for the study')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load initial parameters if provided
    initial_params = None
    if args.initial_params_file:
        with open(args.initial_params_file, 'r') as f:
            initial_params = json.load(f)
    
    study, stats, results_dir = run_optuna_experiment(
        args.experiment, 
        args.n_trials, 
        study_name=args.study_name, 
        use_frobenius_str=args.use_frobenius_str, 
        use_frobenius_rts=args.use_frobenius_rts,
        initial_params=initial_params,
        continue_from=args.continue_from,
        seed=args.seed
    )
    
    print(f"\nOptimization completed. Results saved in {results_dir}")
    print("\nBest trial:")
    if args.experiment == 'mnist':
        best_trial = stats['best_trial']
        print(f"Trial {best_trial['number']}:")
        print(f"  Value: {best_trial['value']:.4f}")
        print("  Parameters:")
        for param, value in best_trial['params'].items():
            print(f"    {param}: {value}")
    else:
        trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
        print(f"Trial {trial['number']}:")
        print(f"  Value: {trial['values'][0]:.4f}")
        print("  Parameters:")
        for param, value in trial['params'].items():
            print(f"    {param}: {value}") 