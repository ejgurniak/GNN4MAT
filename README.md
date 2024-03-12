# GNN4MAT: Graph Neural Networks for Materials

## Run seven different graph neural networks for materials science with one source code

For non-heterogeneous models, go to "Homogeneous models" section below. For heterogeneous models, go to "Heterogeneous models" section below.

## IMPORTANT: make sure all modules for your computer system are properly loaded. This is system specific.

## before you begin: clone the repository from GitHub

```
git clone https://github.com/ejgurniak/CuZr-metallic-glass-GCNNs
```

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

Before running the code, you need to extract the datasets of interest. For the dataset of 1020 samples, that is:

```
tar -xzvf ./datasets/train_1020samples.tar.gz
```

Also copy the code from the code directory to the directory in which you would like to run the code and store the results. For example, the following will copy the python code to the current directory:
```
cp ./code/* .
```

### Homogeneous models: choose which GCNN you want to run. There are 4 options below:

option 1: run the Crystal Edge Graph Attention Neural Network, by Banik et al (https://github.com/sbanik2/CEGANN)

```
python train.py ./train_1020samples model_checkpoints/ log.model GANN
```

option 2: run our Graph Isomorphism Network

```
python train.py ./train_1020samples model_checkpoints log.model GIN
```

option 3: run our GraphSAGE model

```
python train.py ./train_1020samples model_checkpoints log.model SAGE
```

option 4: run our Relational Graph Convolutional Network

```
python train.py ./train_1020samples model_checkpoints log.model RGCN
```

### explanation of the above commands:

train.py: name of the code

./datasets/train_1020samples: path to the training samples. The number of samples in this directory is the number of graphs that the algorithm will load. It is also important to control the train_size and val_size parameters in custom_config.yaml. The first train_size samples are used for training and the last val_size samples are used for validation. Please be careful to make sure you do not overlap and use samples in both training and validation.

model_checkpoints: name of the directory to save the checkpoints, feel free to change

log.model: text file to save training data, feel free to change

RGCN: user-input to determine which neural network to use

## Heterogeneous models: choose which heterogeneous model to run, there are three options below

This code is designed for CuZr alloys. To modify for a different alloy, change hetgraph.py to align with your alloy.

Option 1: Run the Crystal Edge Graph Attention Neural Network in heterogeneous mode. Note: the original model by Banik et al (https://github.com/sbanik2/CEGANN) did not include a heterogeneous mode, this is something we added.

```
python train.py ./train_1020samples model_checkpoints log.model GANN heterogeneous
```

Option 2: Run our GraphSAGE model in heterogeneous mode:
```
python train.py ./train_1020samples model_checkpoints log.model SAGE heterogeneous
```

Option 3: Run our Graph Isomorphism Network in heterogeneous mode:
```
python train.py ./train_1020samples model_checkpoints log.model GIN heterogeneous
```

Note: RGCN is by nature heterogeneous, so we do not run RGCN in heterogeneous mode.

## prediction on the test dataset

```
python predict.py ./target ./model_checkpoints/model_best.pt GIN
```

this will run a prediction job on the homogeneous version of our GIN model

## resume training

If the model training does not converge within the compute time you requested, you can restart the training from a given checkpoint file. To do this, set resume to True in custom_config.yaml. Also copy the checkpoint of choice to ./model_checkpoints/checkpoint.pt. For example, the following will copy the checkpoint from epoch 51 to checkpoint.pt so the model will continue training from epoch 51:
```
cp ./model_checkpoints/checkpoint.pt_51 ./model_checkpoints/checkpoint.pt
```

Replace the "51" in the above command for the epoch from which you would like to resume training
