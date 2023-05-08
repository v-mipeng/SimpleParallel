"""Data utils for Chinese llama
"""


"""GPT2 style dataset."""

import os

import numpy as np
import torch

from utils.utils import print_rank_0


def load_data(data_folder):
    """
    data_prefix (str): "<folder>/*_slice_{i}.npy"
        it will load all npy files matching with the format of data_prefix
    
    """
    files = os.listdir(data_folder)
    data_files = []
    for file in files:
        if file.endswith('npy'):
            data_files.append(file)
    datas = []
    for file in data_files:
        datas.append(np.load(os.path.join(data_folder, file)))
    data = np.concatenate(datas)
    return data


def build_train_valid_test_datasets(data_folder, splits_string,
                                    seq_length):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    data = load_data(data_folder)
    print_rank_0('Load data from {} with {} tokens!'.format(data_folder, len(data)))
    # Print stats about the splits.
    print_rank_0(' > dataset split:')
    # split the data
    values = list(map(float, splits_string.split('-')))
    assert sum(values) == 1.
    if len(values) == 2:
        idx = int(len(data)*values[0])
        train_dataset = GPTDataset(name='train', data=data[:idx], seq_length=seq_length)
        eval_dataset = GPTDataset(name='eval', data=data[idx:], seq_length=seq_length)
        return train_dataset, eval_dataset
    elif len(values) == 3:
        idx = int(len(data)*values[0])
        train_dataset = GPTDataset(name='train', data=data[:idx], seq_length=seq_length)
        end_idx = int(len(data)*(values[0]+values[1]))
        eval_dataset = GPTDataset(name='eval', data=data[idx:end_idx], seq_length=seq_length)
        test_dataset = GPTDataset(name='eval', data=data[end_idx:], seq_length=seq_length)
        return train_dataset, eval_dataset, test_dataset


class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, name, data, seq_length):
        self.name = name
        self.tokens = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) // self.seq_length 

    def __getitem__(self, idx):
        # Get the shuffled index.
        tokens = self.tokens[idx*self.seq_length: (idx+1)*self.seq_length]
        return {'text': np.array(tokens, dtype=np.int64)}