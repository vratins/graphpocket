# graphpocket
Predicting protein binding pocket similarity using graph neural networks.

Code to write:

- processing fpocket to residue pockets -- done
- generating graphs with each node one-hot encoded
- building a dataloader for all the graphs (batch them? tuple them)
- build gvp
- build gcn
- write loss function (contrastive and stability)
- see what scheduling and optimization is needed
- write train.py (with testing) --- dont forget to log runs with wandb

- How do I want to structure it:

- Class to read in the sturctures from a directory
- Class/script to featurize the pockets
    - features would be one-hot encoding (and maybe look at other features that may work like pharmacophoric properties; angles may not make sense when doing an all-atom encoding)
- after featurizing the pocket - generate the graphs, batch them
- gvp module that contains the network, forward pass, backprop -- also contains any code for convolution
- loss file containing contrastive and stability loss
- utils file to read in config and other similar functions
- train file to load in data, hyperparameters, and subsequently train the model
