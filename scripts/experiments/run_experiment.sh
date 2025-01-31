#!/bin/bash

# Take remote configuration as arguments
REMOTE_HOST=$1
REMOTE_PORT=$2
SSH_KEY=$3
REMOTE_DIR=$4
LOCAL_DIR=$5
EXPERIMENT_FILE=$6
shift 6  # Shift to parameter args

# Function to run SSH command and check status
run_ssh_command() {
    ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "$1"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Error: SSH command failed: $1"
        exit $status
    fi
}

# Function to run rsync command and check status
run_rsync_command() {
    rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' \
        --exclude='results/*' --exclude='.pytest_cache' --exclude='*.egg-info' \
        -e "ssh -p $REMOTE_PORT -i $SSH_KEY" "$1" "$2"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "Error: rsync command failed"
        exit $status
    fi
}

echo "Running experiment with parameters: $@"

# Enable strict error handling
set -e  # Exit on any error
set -o pipefail  # Exit if any command in a pipe fails

# Check CUDA version on remote and install appropriate packages
echo "Checking CUDA version on remote..."
CUDA_VERSION=$(ssh -p $REMOTE_PORT -i $SSH_KEY $REMOTE_HOST "nvidia-smi | grep 'CUDA Version' | awk '{print \$9}' | cut -d'.' -f1")
if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA version on remote machine"
    exit 1
fi
echo "Detected CUDA version: $CUDA_VERSION"

# Check number of available GPUs
N_GPUS=$(ssh -p $REMOTE_PORT -i $SSH_KEY $REMOTE_HOST "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l")
echo "Detected $N_GPUS GPUs"

# Install system packages on remote
echo "Installing system packages..."
run_ssh_command "set -e && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip rsync nvidia-cuda-toolkit"

# Ensure remote directory exists
run_ssh_command "set -e && mkdir -p $REMOTE_DIR"

# Sync code to remote with strict error checking
echo "Syncing code..."
run_rsync_command "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

# Install Python packages with proper CUDA support
echo "Installing Python packages..."
run_ssh_command "cd $REMOTE_DIR && \
    python3 -m pip install --upgrade pip && \
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

# Ensure results directory exists and clear it
echo "Clearing remote results directory..."
run_ssh_command "mkdir -p $REMOTE_DIR/results && rm -rf $REMOTE_DIR/results/*"

# Run experiment with GPU
echo "Running experiment..."
if [ "$N_GPUS" -ge 1 ]; then
    run_ssh_command "cd $REMOTE_DIR && CUDA_VISIBLE_DEVICES=0 python3 $EXPERIMENT_FILE $*"
else
    echo "Error: No GPUs detected"
    exit 1
fi

# Get the most recent results directory
LATEST_REMOTE_DIR=$(ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "ls -t $REMOTE_DIR/results | head -n1")

# Create local results directory if it doesn't exist
mkdir -p "$LOCAL_DIR/results/$LATEST_REMOTE_DIR"

# Sync results back
echo "Syncing results back..."
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    "$REMOTE_HOST:$REMOTE_DIR/results/$LATEST_REMOTE_DIR/" \
    "$LOCAL_DIR/results/$LATEST_REMOTE_DIR/"

echo "Done! Results saved in results/$LATEST_REMOTE_DIR/" 