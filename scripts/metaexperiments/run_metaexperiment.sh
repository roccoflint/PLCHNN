#!/bin/bash

# Remote configuration
REMOTE_HOST="root@63.141.33.34"
REMOTE_PORT="22062"
SSH_KEY="~/.ssh/id_ed25519"
REMOTE_DIR="/root/PCHNN"
LOCAL_DIR="/Users/georgeflint/Desktop/research/projects/Hebbian Language/PCHNN"

# Function to generate layer size combinations
generate_layer_sizes() {
    local n_layers=$1
    local sizes=($2)
    local prefix=$3
    local input_size=$4  # New parameter for input size
    
    # Base case: if n_layers is 0, return the prefix
    if [ "$n_layers" -eq 0 ]; then
        echo "$prefix"
        return
    fi
    
    # Recursive case: append each size and recurse
    for size in "${sizes[@]}"; do
        local new_prefix="$prefix,$size"
        if [ "$prefix" = "" ]; then
            new_prefix="$input_size,$size"  # Use provided input size
        fi
        generate_layer_sizes $((n_layers-1)) "$2" "$new_prefix" "$input_size"
    done
}

# Function to generate signific connection combinations
generate_signific_connections() {
    local n_layers=$1
    local prefix=$2
    local result=()
    
    # Base case: if n_layers is 0, return the prefix
    if [ "$n_layers" -eq 0 ]; then
        # Ensure at least one True
        if [[ "$prefix" != *"True"* ]]; then
            return
        fi
        echo "$prefix"
        return
    fi
    
    # Recursive case: try both True and False
    for val in "True" "False"; do
        local new_prefix="$prefix,$val"
        if [ "$prefix" = "" ]; then
            new_prefix="$val"
        fi
        generate_signific_connections $((n_layers-1)) "$new_prefix"
    done
}

# Define parameter grids
declare -a EXPERIMENT_TYPES=("colored_mnist")  # Only colored_mnist
declare -a N_COLORS=(2)  # 2 colors (easy difficulty)
declare -a N_LAYERS=(1)  # Single layer
declare -a LAYER_SIZES=(400)  # Size 400
declare -a PATHWAY_INTERACTION=("False")  # No pathway interaction

# # Check and install rsync on remote if needed
# echo "Checking rsync installation on remote..."
# ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "which rsync || (apt-get update && apt-get install -y rsync)"

# # Create remote directory structure
# echo "Creating remote directory structure..."
# ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "mkdir -p $REMOTE_DIR/results"

# Sync code to remote first
echo "Syncing code to remote..."
rsync -avz -e "ssh -p $REMOTE_PORT -i $SSH_KEY" \
    --exclude "results" \
    --exclude "__pycache__" \
    "$LOCAL_DIR/" \
    "$REMOTE_HOST:$REMOTE_DIR/"

# Now install Python requirements
echo "Installing Python requirements on remote..."
ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "cd $REMOTE_DIR && pip install -r requirements.txt"

# Get available GPUs on remote
echo "Checking available GPUs on remote..."
N_GPUS=$(ssh $REMOTE_HOST -p $REMOTE_PORT -i $SSH_KEY "nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l")

if [ $N_GPUS -eq 0 ]; then
    echo "No GPUs found on remote, will run on CPU"
    N_GPUS=1
else
    echo "Found $N_GPUS GPUs on remote"
fi

# Create a temporary directory for job files
TEMP_DIR="scripts/metaexperiments/temp_jobs"
mkdir -p $TEMP_DIR

# Generate all parameter combinations and create job files
JOB_INDEX=0
for exp_type in "${EXPERIMENT_TYPES[@]}"; do
    if [ "$exp_type" = "mnist" ]; then
        input_size=784
        experiment_file="scripts/experiments/mnist.py"
    else
        for n_colors in "${N_COLORS[@]}"; do
            input_size=$((784 * (1 + n_colors)))  # Input size depends on number of colors
            experiment_file="scripts/experiments/colored_mnist.py"
            
            for n_layer in "${N_LAYERS[@]}"; do
                # Generate layer size combinations
                while IFS= read -r layer_sizes; do
                    # Generate signific connection combinations
                    while IFS= read -r signific_connections; do
                        for pathway in "${PATHWAY_INTERACTION[@]}"; do
                            # Create job file
                            JOB_FILE="$TEMP_DIR/job_${JOB_INDEX}.sh"
                            echo "#!/bin/bash" > $JOB_FILE
                            echo "export CUDA_VISIBLE_DEVICES=$((JOB_INDEX % N_GPUS))" >> $JOB_FILE
                            echo "bash scripts/experiments/run_experiment.sh \"$REMOTE_HOST\" \"$REMOTE_PORT\" \"$SSH_KEY\" \"$REMOTE_DIR\" \"$LOCAL_DIR\" \"$experiment_file\" --n_colors=$n_colors --hidden_size=${layer_sizes##*,} --allow_pathway_interaction=$pathway --n_epochs=20" >> $JOB_FILE
                            chmod +x $JOB_FILE
                            
                            # Run job in background
                            ./$JOB_FILE &
                            
                            # If we've filled all GPUs, wait for one to finish
                            if [ $((JOB_INDEX % N_GPUS)) -eq $((N_GPUS - 1)) ]; then
                                wait
                            fi
                            
                            JOB_INDEX=$((JOB_INDEX + 1))
                        done
                    done < <(generate_signific_connections $n_layer "")
                done < <(generate_layer_sizes $n_layer "${LAYER_SIZES[*]}" "" "$input_size")
            done
        done
    fi
done

# Wait for remaining jobs
wait

# Cleanup
rm -rf $TEMP_DIR 