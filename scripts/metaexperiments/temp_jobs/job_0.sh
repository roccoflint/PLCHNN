#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/experiments/run_experiment.sh "root@63.141.33.34" "22062" "~/.ssh/id_ed25519" "/root/PCHNN" "/Users/georgeflint/Desktop/research/projects/Hebbian Language/PCHNN" "scripts/experiments/colored_mnist.py" --n_colors=2 --hidden_size=400 --allow_pathway_interaction=False --n_epochs=20
