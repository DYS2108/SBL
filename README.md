# Unsupervised Deep Spectral Basis Learning for Generalized Eigendecomposition and Spectral Embedding
Implementation of paper - towards an unsupervised spectral basis learning framework for generalized eigendecomposition of graph matrices.

## Setup
The code can be run under any environment with >= Python 3.6 and >= tensorflow1.15.4.

Install the following packages:
```
pip install -r requirements.txt
```
## Dataset
The download path of public available dataset:
FAUST: https://faust-leaderboard.is.tuebingen.mpg.de
SCAPE: https://ai.stanford.edu/~drago/Projects/scape/scape.html
TOSCA: https://cis.cs.technion.ac.ildata/toscahires.txt

## Training
We provide the training code. You can run the main.py based on specified `dataset_path`, `results_path`, and set IS_TRAIN as 1.

## Testing
You can run the main.py with IS_TRAIN = 0 to get the embedding of testing dataset.
