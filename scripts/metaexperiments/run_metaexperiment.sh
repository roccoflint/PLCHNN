#!/bin/bash

# Set remote connection details
REMOTE_HOST="213.173.108.40"
REMOTE_PORT="39469"
SSH_KEY="~/.ssh/id_ed25519"

# Parse arguments
N_TRIALS=20  # Fixed number of trials for each study

# Pre-generate timestamps for new studies
TIMESTAMP_BASE=$(date +%Y%m%d_%H%M%S)
TIMESTAMP_COLORED_FROB="${TIMESTAMP_BASE}_01"
TIMESTAMP_COLORED_RAW="${TIMESTAMP_BASE}_02"

# Save initial parameters for new studies
cat > initial_params.json << EOL
{
    "hidden_size": 136,
    "p": 2.4090317051787107,
    "k": 3,
    "delta": 0.22118115364137012,
    "signific_p_multiplier": 2.4096762883665344,
    "allow_pathway_interaction": true,
    "n_epochs": 22,
    "batch_size": 106,
    "learning_rate": 0.022182236416172403
}
EOL

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

if [ "$N_GPUS" -lt 4 ]; then
    echo "Error: This script requires 4 GPUs, but only $N_GPUS found"
    exit 1
fi

# Enable strict error handling
set -e
set -o pipefail

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
rsync -avz --exclude 'results' --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '.pytest_cache' --exclude '*.egg-info' -e "ssh -p $REMOTE_PORT -i $SSH_KEY" . root@$REMOTE_HOST:/root/PCHNN/ || {
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

echo "Running four parallel studies..."

# 1. Continue MNIST study with Frobenius norm
(
    echo "Continuing MNIST Optuna experiment with Frobenius norm on GPU 0..."
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=0 python3 scripts/metaexperiments/optuna_experiment.py --experiment mnist --n_trials $N_TRIALS --use_frobenius --continue_from results/optuna_mnist_optimization_20250130_203813_01/study.db"
) &

# 2. Continue colored MNIST study with raw accuracy
(
    echo "Continuing colored MNIST Optuna experiment with raw accuracy on GPU 1..."
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=1 python3 scripts/metaexperiments/optuna_experiment.py --experiment colored_mnist --n_trials $N_TRIALS --continue_from results/optuna_colored_mnist_optimization_20250130_203813_04/study.db"
) &

# 3. Start new colored MNIST with raw accuracy and initial parameters
(
    echo "Starting new colored MNIST Optuna experiment with raw accuracy and initial parameters on GPU 2..."
    STUDY_NAME="optuna_colored_mnist_optimization_${TIMESTAMP_COLORED_RAW}"
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" initial_params.json root@$REMOTE_HOST:/root/PCHNN/
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=2 python3 scripts/metaexperiments/optuna_experiment.py --experiment colored_mnist --n_trials $N_TRIALS --study_name $STUDY_NAME --initial_params_file initial_params.json"
) &

# 4. Start new colored MNIST with Frobenius norm and initial parameters
(
    echo "Starting new colored MNIST Optuna experiment with Frobenius norm and initial parameters on GPU 3..."
    STUDY_NAME="optuna_colored_mnist_optimization_${TIMESTAMP_COLORED_FROB}"
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && CUDA_VISIBLE_DEVICES=3 python3 scripts/metaexperiments/optuna_experiment.py --experiment colored_mnist --n_trials $N_TRIALS --use_frobenius --study_name $STUDY_NAME --initial_params_file initial_params.json"
) &

# Wait for all experiments to complete
wait

# Sync results back
echo "Syncing results back..."
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    root@$REMOTE_HOST:/root/PCHNN/results/optuna_*/ \
    results/

# Clean up
rm initial_params.json

echo "Done! Results saved in results directory."