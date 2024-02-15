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
