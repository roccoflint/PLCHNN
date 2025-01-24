#!/bin/bash

# Remote configuration
REMOTE_HOST="root@69.30.85.115"
REMOTE_PORT="22054"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/root/PCHNN"
LOCAL_DIR="/Users/georgeflint/Desktop/research/projects/Hebbian Language/PCHNN"

# Check and install rsync on remote if needed
echo "Checking rsync installation on remote..."
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "which rsync || (apt-get update && apt-get install -y rsync)"

# Create remote directory structure
echo "Creating remote directory structure..."
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "mkdir -p $REMOTE_DIR/results"

# Sync code to remote
echo "Syncing code to remote..."
rsync -az -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    --exclude "results" \
    --exclude "__pycache__" \
    --exclude ".git" \
    "$LOCAL_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/"

# Install Python requirements
echo "Installing Python requirements on remote..."
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "cd $REMOTE_DIR && pip install -r requirements.txt"

# Check GPUs
echo "Checking available GPUs on remote..."
N_GPUS=$(ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l")
echo "Found $N_GPUS GPUs on remote"

# Run experiment
echo "Running experiment with parameters: $@"
python3 scripts/experiments/colored_mnist.py $@

# Sync results back
echo "Syncing results back..."
rsync -az -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    "$REMOTE_HOST:$REMOTE_DIR/results/" \
    "$LOCAL_DIR/results/" 