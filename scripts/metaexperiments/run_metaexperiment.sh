#!/bin/bash

# Parse command line arguments
RUN_MNIST=0
RUN_COLORED_MNIST=0
CONTINUE_FROM=""

print_usage() {
    echo "Usage: $0 [--mnist] [--colored-mnist] [--continue_from path/to/study.db]"
    echo "At least one experiment must be specified"
    exit 1
}

# Parse arguments
[ $# -eq 0 ] && print_usage
while [ $# -gt 0 ]; do
    case "$1" in
        --mnist) RUN_MNIST=1 ;;
        --colored-mnist) RUN_COLORED_MNIST=1 ;;
        --continue_from) 
            shift
            CONTINUE_FROM="$1" ;;
        *) print_usage ;;
    esac
    shift
done

# Set remote connection details
REMOTE_HOST="213.173.110.202"
REMOTE_PORT="17227"
SSH_KEY="~/.ssh/id_ed25519"

# Set trial count
MNIST_TRIALS=20
COLORED_MNIST_TRIALS=20

# Set study names
if [ $RUN_MNIST -eq 1 ]; then
    MNIST_STUDY="optuna_mnist_single_objective_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "results/${MNIST_STUDY}"
    cp scripts/metaexperiments/initial_params.json initial_params_mnist.json
fi

if [ $RUN_COLORED_MNIST -eq 1 ]; then
    COLORED_MNIST_STUDY="optuna_colored_mnist_single_objective_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "results/${COLORED_MNIST_STUDY}"
    cp scripts/metaexperiments/initial_params.json initial_params_colored_mnist.json
fi

# Enable strict error handling
set -e
set -o pipefail

# Install system packages on remote
echo "Installing system packages..."
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "set -e && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip rsync nvidia-cuda-toolkit"

# Create project directory and sync code
echo "Setting up remote environment..."
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "set -e && mkdir -p /root/PCHNN"

# Sync code
echo "Syncing code..."
rsync -avz --exclude 'results' --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' --exclude '.pytest_cache' --exclude '*.egg-info' -e "ssh -p $REMOTE_PORT -i $SSH_KEY" . root@$REMOTE_HOST:/root/PCHNN/ || {
    echo "Error: Failed to sync code"
    exit 1
}

# Install Python packages
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

# Create remote directories with proper permissions
echo "Creating remote directories..."
REMOTE_DIRS_CMD="chmod -R 777 /root/PCHNN/results"
if [ $RUN_MNIST -eq 1 ]; then
    REMOTE_DIRS_CMD="mkdir -p \"/root/PCHNN/results/${MNIST_STUDY}\" && $REMOTE_DIRS_CMD"
fi
if [ $RUN_COLORED_MNIST -eq 1 ]; then
    REMOTE_DIRS_CMD="mkdir -p \"/root/PCHNN/results/${COLORED_MNIST_STUDY}\" && $REMOTE_DIRS_CMD"
fi
ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "$REMOTE_DIRS_CMD"

# Copy initial parameters to remote
PARAMS_TO_COPY=""
if [ $RUN_MNIST -eq 1 ]; then
    PARAMS_TO_COPY="initial_params_mnist.json"
fi
if [ $RUN_COLORED_MNIST -eq 1 ]; then
    PARAMS_TO_COPY="$PARAMS_TO_COPY initial_params_colored_mnist.json"
fi
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" $PARAMS_TO_COPY "root@${REMOTE_HOST}:/root/PCHNN/"

# Run experiments
if [ $RUN_MNIST -eq 1 ]; then
    echo "Running MNIST experiment..."
    CONTINUE_ARG=""
    if [ ! -z "$CONTINUE_FROM" ]; then
        CONTINUE_ARG="--continue_from \"$CONTINUE_FROM\""
    fi
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && \
        CUDA_VISIBLE_DEVICES=0 python3 scripts/metaexperiments/optuna_experiment.py \
            --experiment mnist \
            --n_trials $MNIST_TRIALS \
            --initial_params_file initial_params_mnist.json \
            --study_name \"${MNIST_STUDY}\" \
            --use_frobenius_str \
            --use_frobenius_rts \
            $CONTINUE_ARG"
fi

if [ $RUN_COLORED_MNIST -eq 1 ]; then
    echo "Running Colored MNIST experiment..."
    CONTINUE_ARG=""
    if [ ! -z "$CONTINUE_FROM" ]; then
        CONTINUE_ARG="--continue_from \"$CONTINUE_FROM\""
    fi
    ssh -p $REMOTE_PORT -i $SSH_KEY root@$REMOTE_HOST "cd /root/PCHNN && \
        CUDA_VISIBLE_DEVICES=0 python3 scripts/metaexperiments/optuna_experiment.py \
            --experiment colored_mnist \
            --n_trials $COLORED_MNIST_TRIALS \
            --initial_params_file initial_params_colored_mnist.json \
            --study_name \"${COLORED_MNIST_STUDY}\" \
            --use_frobenius_rts \
            $CONTINUE_ARG"
fi

# Sync results back
echo "Syncing results back..."
if [ $RUN_MNIST -eq 1 ]; then
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@${REMOTE_HOST}:/root/PCHNN/results/${MNIST_STUDY}/" \
        "results/${MNIST_STUDY}/"
fi
if [ $RUN_COLORED_MNIST -eq 1 ]; then
    rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
        "root@${REMOTE_HOST}:/root/PCHNN/results/${COLORED_MNIST_STUDY}/" \
        "results/${COLORED_MNIST_STUDY}/"
fi

# Clean up
if [ $RUN_MNIST -eq 1 ]; then
    rm initial_params_mnist.json
fi
if [ $RUN_COLORED_MNIST -eq 1 ]; then
    rm initial_params_colored_mnist.json
fi

echo "Done! Results saved in results directory."