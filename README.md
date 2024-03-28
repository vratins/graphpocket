# graphpocket
Predicting protein binding pocket similarity using the Geometric Vector Perceptron - GNN architecture (insert link)

- Config files: Model + Data + WandB parameters (+hyperparameter sweep ideas)

- General Directory Layout:
    - GraphPocket.py -- reads in a pocket and generates a DGL graph object
    - Dataloader.py -- Creates a tuple dataset along with labels, as well as creates sequence cluster splits
    - Model.py -- GVP, GVPEdgeConv, ReceptorEncoder classes + loss function(s)
    - Train.py -- Training and logging
    - utils.py
