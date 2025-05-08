# Primitive Linguistic Compositionality in a Hebbian Neural Network

This repository contains the code required to replicate the experiments in the titular paper published in (Proceedings of the Annual Meeting of the Cognitive Science Society)[https://escholarship.org/uc/cognitivesciencesociety].

## Warning

This work was exploratory in nature. The codebase was thus developed as a playground rather than a tightly-maintained repository. Instructions for replicating experiments are provided, but the codebase itself is a bit clunky, and a detailed guide to it is not presented in this readme. For questions and/or correspondence, please reach out to georgeflint@berkeley.edu.

This work is part of a broader research program of the first author. If a future work in this program uses these or highly similar rules, it will be linked here.

## Quickstart

To replicate Experiment 1 (noncompositional MNIST), clone the repository and run the following command:

```
python scripts/experiments/mnist.py \
    --hidden_size 279 \
    --p 2.7149 \
    --k 2 \
    --delta 0.2847 \
    --signific_p_multiplier 3.7437 \
    --allow_pathway_interaction false \
    --n_epochs 44 \
    --batch_size 115 \
    --learning_rate 0.0240
```

To replicate Experiment 2 (compositional Colored MNIST), clone the repository and run the following command:

```
python scripts/experiments/colored_mnist.py \
    --hidden_size 107 \
    --n_colors 5 \
    --p 3.272820822527561 \
    --delta 0.4034282764658811 \
    --k 3 \
    --allow_pathway_interaction false \
    --signific_p_multiplier 3.722699421778279 \
    --n_epochs 48 \
    --batch_size 78 \
    --learning_rate 0.011227530419758448 \
    --train_ratio 0.8
```

**Results will appear in a timestamp directory in the `/results/` directory.**