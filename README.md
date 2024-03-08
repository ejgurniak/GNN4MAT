# CuZr-metallic-glass-GCNNs

# Run four different neural networks with one source code


## IMPORTANT: make sure all modules for your computer system are properly loaded. This is system specific.

## environment creation

step 1: 

```
conda env create --name ejg-alt -f alt_environment.yml
```

If the above does not work, try: mamba env create --name ejg-alt -f alt_environment.yml

step 2:

```
conda activate ejg-alt
```

step 3:

```
pip install torch_geometric
```

## running the code

Note: our code builds on the code found here: https://github.com/sbanik2/CEGANN

option 1: run the Crystal Edge Graph Attention Neural Network, by Banik et al

```
python train.py ./train model_checkpoints/ log.model GANN
```

option 2: run our Graph Isomorphism Network

```
python train.py ./train model_checkpoints log.model GIN
```

option 3: run our GraphSAGE model

```
python train.py ./train model_checkpoints log.model SAGE
```

option 4: run our Relational Graph Convolutional Network

```
python train.py ./train model_checkpoints log.model RGCN
```

### explanation of the above commands:

train.py: name of the code

./train: path to the training samples. The number of samples in this directory is the number of graphs that the algorithm will load. It is also important to control the train_size and val_size parameters in custom_config.yaml. The first train_size samples are used for training and the last val_size samples are used for validation. Please be careful to make sure you do not overlap and use samples in both training and validation.

model_checkpoints: name of the directory to save the checkpoints, feel free to change

log.model: text file to save training data, feel free to change

RGCN: user-input to determine which neural network to use

## heterogeneous mode

to run a heterogeneous version of a model, add the keyword "heterogeneous" to the command to run the script

for example:

```
python train.py ./train model_checkpoints log.model SAGE heterogeneous
```

this will run the heterogeneous version of our GraphSAGE model

## prediction on the test dataset

```
python predict.py ./target ./model_checkpoints/model_best.pt GIN
```

this will run a prediction job on the homogeneous version of our GIN model
