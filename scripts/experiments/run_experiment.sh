#!/bin/bash

# Take remote configuration as arguments
REMOTE_HOST=$1
REMOTE_PORT=$2
SSH_KEY=$3
REMOTE_DIR=$4
LOCAL_DIR=$5
EXPERIMENT_FILE=$6
shift 6  # Shift to parameter args

# Run experiment with provided parameters
echo "Running experiment with parameters: $@"

# Clear remote results directory
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "rm -rf $REMOTE_DIR/results/*"

# Run experiment (properly quote arguments)
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "cd $REMOTE_DIR && python3 $EXPERIMENT_FILE $*"

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