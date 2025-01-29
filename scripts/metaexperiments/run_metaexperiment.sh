#!/bin/bash

# Set remote connection details
REMOTE_HOST="194.68.245.28"
REMOTE_PORT="22079"
SSH_KEY="~/.ssh/id_ed25519"

# Parse arguments
CONTINUE_MNIST=""
CONTINUE_COLORED=""
N_TRIALS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --continue-mnist)
            CONTINUE_MNIST="$2"
            shift 2
            ;;
        --continue-colored)
            CONTINUE_COLORED="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create results directory locally
mkdir -p results

# Check CUDA version on remote and install appropriate packages
echo "Checking CUDA version on remote..."
CUDA_VERSION=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "nvidia-smi | grep 'CUDA Version' | awk '{print \$9}' | cut -d'.' -f1")
if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA version on remote machine"
    exit 1
fi
echo "Detected CUDA version: $CUDA_VERSION"

# Check number of available GPUs
N_GPUS=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l")
echo "Detected $N_GPUS GPUs"

# Enable strict error handling
set -e  # Exit on any error
set -o pipefail  # Exit if any command in a pipe fails

# Install system packages on remote
echo "Installing system packages..."
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "set -e && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip rsync nvidia-cuda-toolkit"

# Create project directory and sync code first
echo "Setting up remote environment..."
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "set -e && mkdir -p /root/PCHNN"

# Sync with strict error checking
echo "Syncing code..."
rsync -avz --exclude 'results' --exclude '.git' -e "ssh -p $REMOTE_PORT -i $SSH_KEY" . root@$REMOTE_HOST:/root/PCHNN/ || {
    echo "Error: Failed to sync code"
    exit 1
}

# Install Python packages with strict verification
echo "Installing Python packages..."
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir optuna && \
    python3 -m pip install -r requirements.txt && \
    CUDA_VER=\$(nvidia-smi | grep 'CUDA Version' | awk '{print \$9}' | cut -d'.' -f1) && \
    if [ \"\$CUDA_VER\" -ge 12 ]; then \
        python3 -m pip install --no-cache-dir cupy-cuda12x; \
    elif [ \"\$CUDA_VER\" -ge 11 ]; then \
        python3 -m pip install --no-cache-dir cupy-cuda11x; \
else \
        echo \"Error: Unsupported CUDA version \$CUDA_VER (need 11 or higher)\"; \
    exit 1; \
fi"

# Create results directory on remote
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "mkdir -p /root/PCHNN/results"

# Copy study databases if continuing from previous studies
if [ -n "$CONTINUE_MNIST" ]; then
    echo "Copying MNIST study database to remote..."
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "mkdir -p /root/PCHNN/$CONTINUE_MNIST"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "$CONTINUE_MNIST/study.db" \
        root@$REMOTE_HOST:/root/PCHNN/$CONTINUE_MNIST/
fi

if [ -n "$CONTINUE_COLORED" ]; then
    echo "Copying colored MNIST study database to remote..."
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "mkdir -p /root/PCHNN/$CONTINUE_COLORED"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "$CONTINUE_COLORED/study.db" \
        root@$REMOTE_HOST:/root/PCHNN/$CONTINUE_COLORED/
fi

if [ "$N_GPUS" -ge 2 ]; then
    echo "Running experiments in parallel on separate GPUs..."
    
    # Run MNIST on GPU 0 first
    (
        echo "Running MNIST Optuna experiment on GPU 0..."
        CONTINUE_ARG=""
        if [ -n "$CONTINUE_MNIST" ]; then
            echo "Continuing from previous MNIST study: $CONTINUE_MNIST"
            CONTINUE_ARG="--continue_from $CONTINUE_MNIST"
        fi
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=0 python3 scripts/metaexperiments/optuna_experiment.py --experiment mnist --n_trials $N_TRIALS $CONTINUE_ARG"
    ) &

    # Wait a few seconds to ensure different timestamps
    sleep 3
    
    # Then run colored MNIST on GPU 1
    (
        echo "Running colored MNIST Optuna experiment on GPU 1..."
        CONTINUE_ARG=""
        if [ -n "$CONTINUE_COLORED" ]; then
            echo "Continuing from previous colored MNIST study: $CONTINUE_COLORED"
            CONTINUE_ARG="--continue_from $CONTINUE_COLORED"
        fi
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=1 python3 scripts/metaexperiments/optuna_experiment.py --experiment colored_mnist --n_trials $N_TRIALS $CONTINUE_ARG"
    ) &

    # Wait for both experiments to complete
    wait

    # Get latest Optuna results directories
    LATEST_OPTUNA_MNIST=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_mnist | head -n1")
    LATEST_OPTUNA_COLORED=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_colored_mnist | head -n1")

    # Create local directories and sync Optuna results
    echo "Syncing Optuna results back..."
    mkdir -p "results/$LATEST_OPTUNA_MNIST"
    mkdir -p "results/$LATEST_OPTUNA_COLORED"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_OPTUNA_MNIST/ \
        "results/$LATEST_OPTUNA_MNIST/"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_OPTUNA_COLORED/ \
        "results/$LATEST_OPTUNA_COLORED/"

    # Extract best parameters for MNIST
    echo "Extracting best MNIST parameters..."
    MNIST_PARAMS=$(python3 -c "import json
with open('results/$LATEST_OPTUNA_MNIST/study_stats.json') as f:
    stats = json.load(f)
best_trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
params = best_trial['params']
param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
print(param_str)")

    # Extract best parameters for colored MNIST
    echo "Extracting best colored MNIST parameters..."
    COLORED_PARAMS=$(python3 -c "import json
with open('results/$LATEST_OPTUNA_COLORED/study_stats.json') as f:
    stats = json.load(f)
best_trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
params = best_trial['params']
param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
print(param_str)")

    # Run best trials in parallel if multiple GPUs
    if [ "$N_GPUS" -ge 2 ]; then
        echo "Running best trials in parallel..."
        
        # Run MNIST best trial first
        (
            echo "Running MNIST best trial on GPU 0..."
            ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=0 python3 scripts/experiments/mnist.py $MNIST_PARAMS"
        ) &

        # Wait a few seconds to ensure different timestamps
        sleep 3
        
        # Then run colored MNIST best trial
        (
            echo "Running colored MNIST best trial on GPU 1..."
            ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=1 python3 scripts/experiments/colored_mnist.py $COLORED_PARAMS --n_colors 2 --train_ratio 0.8"
        ) &

        wait

        # Get latest non-optuna results directories for best trials
        echo "Looking for best trial results..."
        sleep 5  # Give filesystem time to update
        
        # Find and sync MNIST results
        echo "Looking for MNIST best trial..."
        LATEST_MNIST=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -t results | grep -v optuna | grep mnist_best | head -n1")
        echo "Debug: Found MNIST best trial directory: $LATEST_MNIST"
        
        if [ -z "$LATEST_MNIST" ]; then
            echo "Error: Could not find MNIST best trial results directory"
            ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -lt results"
            exit 1
        fi
        
        echo "Syncing MNIST best trial results..."
        mkdir -p "results/$LATEST_MNIST"
        rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
            root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_MNIST/ \
            "results/$LATEST_MNIST/"
            
        # Find and sync Colored MNIST results
        echo "Looking for Colored MNIST best trial..."
        LATEST_COLORED=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -t results | grep -v optuna | grep colored_best | head -n1")
        echo "Debug: Found Colored MNIST best trial directory: $LATEST_COLORED"
        
        if [ -z "$LATEST_COLORED" ]; then
            echo "Error: Could not find Colored MNIST best trial results directory"
            ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -lt results"
            exit 1
        fi
        
        echo "Syncing Colored MNIST best trial results..."
        mkdir -p "results/$LATEST_COLORED"
        rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
            root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_COLORED/ \
            "results/$LATEST_COLORED/"

        echo "Done! Results saved in:"
        echo "- MNIST Optuna: results/$LATEST_OPTUNA_MNIST/"
        echo "- MNIST Best: results/$LATEST_MNIST/"
        echo "- Colored MNIST Optuna: results/$LATEST_OPTUNA_COLORED/"
        echo "- Colored MNIST Best: results/$LATEST_COLORED/"
    else
        echo "Running best trials sequentially..."
        echo "Running MNIST best trial..."
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/experiments/mnist.py $MNIST_PARAMS"
        
        echo "Running colored MNIST best trial..."
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/experiments/colored_mnist.py $COLORED_PARAMS --n_colors 2 --train_ratio 0.8"
    fi
else
    echo "Running experiments sequentially on single GPU..."
    # Run MNIST Optuna experiment
    echo "Running MNIST Optuna experiment..."
    CONTINUE_ARG=""
    if [ -n "$CONTINUE_MNIST" ]; then
        echo "Continuing from previous MNIST study: $CONTINUE_MNIST"
        CONTINUE_ARG="--continue_from $CONTINUE_MNIST"
    fi
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/metaexperiments/optuna_experiment.py --experiment mnist --n_trials $N_TRIALS $CONTINUE_ARG"

    # Get latest Optuna results directory
    LATEST_OPTUNA_MNIST=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_mnist | head -n1")

    # Create local directory and sync Optuna results
    echo "Syncing MNIST Optuna results back..."
    mkdir -p "results/$LATEST_OPTUNA_MNIST"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_OPTUNA_MNIST/ \
        "results/$LATEST_OPTUNA_MNIST/"

    # Extract best parameters for MNIST
    echo "Extracting best MNIST parameters..."
    MNIST_PARAMS=$(python3 -c "import json
with open('results/$LATEST_OPTUNA_MNIST/study_stats.json') as f:
    stats = json.load(f)
best_trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
params = best_trial['params']
param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
print(param_str)")

    # Run MNIST experiment with best parameters
    echo "Running MNIST experiment with best parameters..."
    echo "Using parameters: $MNIST_PARAMS"
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/experiments/mnist.py $MNIST_PARAMS"

    # Find and sync MNIST results
    echo "Looking for MNIST best trial..."
    sleep 5  # Give filesystem time to update
    LATEST_MNIST=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -t results | grep -v optuna | grep mnist_best | head -n1")
    echo "Debug: Found MNIST best trial directory: $LATEST_MNIST"
    
    if [ -z "$LATEST_MNIST" ]; then
        echo "Error: Could not find MNIST best trial results directory"
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -lt results"
        exit 1
    fi
    
    echo "Syncing MNIST best trial results..."
    mkdir -p "results/$LATEST_MNIST"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_MNIST/ \
        "results/$LATEST_MNIST/"

    echo "Done! Results saved in:"
    echo "- MNIST Optuna: results/$LATEST_OPTUNA_MNIST/"
    echo "- MNIST Best: results/$LATEST_MNIST/"

# Run colored MNIST Optuna experiment
echo "Running colored MNIST Optuna experiment..."
    CONTINUE_ARG=""
    if [ -n "$CONTINUE_COLORED" ]; then
        echo "Continuing from previous colored MNIST study: $CONTINUE_COLORED"
        CONTINUE_ARG="--continue_from $CONTINUE_COLORED"
    fi
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/metaexperiments/optuna_experiment.py --experiment colored_mnist --n_trials $N_TRIALS $CONTINUE_ARG"

# Get latest Optuna results directory
LATEST_OPTUNA_COLORED=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_colored_mnist | head -n1")

# Create local directory and sync Optuna results
echo "Syncing colored MNIST Optuna results back..."
mkdir -p "results/$LATEST_OPTUNA_COLORED"
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_OPTUNA_COLORED/ \
    "results/$LATEST_OPTUNA_COLORED/"

# Extract best parameters for colored MNIST
echo "Extracting best colored MNIST parameters..."
    COLORED_PARAMS=$(python3 -c "import json
with open('results/$LATEST_OPTUNA_COLORED/study_stats.json') as f:
    stats = json.load(f)
best_trial = min(stats['pareto_front'], key=lambda x: x['values'][0])
params = best_trial['params']
param_str = ' '.join([f'--{k} {v}' for k, v in params.items()])
print(param_str)")

# Run colored MNIST experiment with best parameters
echo "Running colored MNIST experiment with best parameters..."
    echo "Using parameters: $COLORED_PARAMS"
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && python3 scripts/experiments/colored_mnist.py $COLORED_PARAMS --n_colors 2 --train_ratio 0.8"

    # Find and sync Colored MNIST results
    echo "Looking for Colored MNIST best trial..."
    sleep 5  # Give filesystem time to update
    LATEST_COLORED=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -t results | grep -v optuna | grep colored_best | head -n1")
    echo "Debug: Found Colored MNIST best trial directory: $LATEST_COLORED"
    
    if [ -z "$LATEST_COLORED" ]; then
        echo "Error: Could not find Colored MNIST best trial results directory"
        ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && ls -lt results"
        exit 1
    fi
    
    echo "Syncing Colored MNIST best trial results..."
mkdir -p "results/$LATEST_COLORED"
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_COLORED/ \
    "results/$LATEST_COLORED/"

echo "Done! Results saved in:"
echo "- Colored MNIST Optuna: results/$LATEST_OPTUNA_COLORED/"
echo "- Colored MNIST Best: results/$LATEST_COLORED/"
fi

# After experiments complete, sync all results back
echo "Syncing all results back to local machine..."

if [ -n "$CONTINUE_MNIST" ]; then
    echo "Syncing MNIST results..."
    mkdir -p "$CONTINUE_MNIST"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@$REMOTE_HOST:/root/PCHNN/$CONTINUE_MNIST/" \
        "$CONTINUE_MNIST/"
fi

if [ -n "$CONTINUE_COLORED" ]; then
    echo "Syncing colored MNIST results..."
    mkdir -p "$CONTINUE_COLORED"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@$REMOTE_HOST:/root/PCHNN/$CONTINUE_COLORED/" \
        "$CONTINUE_COLORED/"
fi

# Get latest results directories from remote
LATEST_MNIST=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_mnist | head -n1")
LATEST_COLORED=$(ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "ls -t /root/PCHNN/results | grep optuna_colored_mnist | head -n1")

if [ -n "$LATEST_MNIST" ]; then
    echo "Syncing latest MNIST results..."
    mkdir -p "results/$LATEST_MNIST"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_MNIST/" \
        "results/$LATEST_MNIST/"
fi

if [ -n "$LATEST_COLORED" ]; then
    echo "Syncing latest colored MNIST results..."
    mkdir -p "results/$LATEST_COLORED"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@$REMOTE_HOST:/root/PCHNN/results/$LATEST_COLORED/" \
        "results/$LATEST_COLORED/"
fi

echo "Done! Results saved in:"
if [ -n "$LATEST_MNIST" ]; then
    echo "- MNIST Optuna: results/$LATEST_MNIST/"
fi
if [ -n "$LATEST_COLORED" ]; then
    echo "- Colored MNIST Optuna: results/$LATEST_COLORED/"
fi