#!/bin/bash

# Run all experiments
echo "Running experiment 1: 30 epochs, pathway interaction true"
bash scripts/metaexperiments/run_metaexperiment.sh --n_colors=2 --hidden_size=324 --allow_pathway_interaction=True --n_epochs=30

echo "Running experiment 2: 30 epochs, pathway interaction false"
bash scripts/metaexperiments/run_metaexperiment.sh --n_colors=2 --hidden_size=324 --allow_pathway_interaction=False --n_epochs=30

echo "Running experiment 3: 50 epochs, pathway interaction true"
bash scripts/metaexperiments/run_metaexperiment.sh --n_colors=2 --hidden_size=324 --allow_pathway_interaction=True --n_epochs=50

echo "Running experiment 4: 50 epochs, pathway interaction false"
bash scripts/metaexperiments/run_metaexperiment.sh --n_colors=2 --hidden_size=324 --allow_pathway_interaction=False --n_epochs=50 