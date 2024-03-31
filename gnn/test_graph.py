import dgl
import pickle
import torch

from dataloader import (create_dataset, get_dataloader)


def main():

    pos_path, neg_path = '../data/TOUGH-M1_positive.list', '../data/TOUGH-M1_negative.list'

    train_dataset, test_dataset = create_dataset(pos_path, neg_path, fold_nr=0, type='seq')
    train_dataloader = get_dataloader(train_dataset, batch_size=5, num_workers=0)

    for batch in train_dataloader:
        print(batch)
        # print(f"Batch size: {len(labels)}")
        # print(f"Labels: {labels}")
        break

if __name__=='__main__':
    main()
