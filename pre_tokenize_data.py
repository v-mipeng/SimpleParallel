""" Tokenize the text files and serialize it.

1. We plan to save n tokens a file.
2. Different process of data parallel group may load different file
"""

import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, default="/home/deepspeed/projects/PLLM/data/test.txt",
                       help='Path to input JSON')
    group.add_argument('--json-key', nargs='+', default='text',
                       help='space separate key to extract from json')

    group = parser.add_argument_group(title='tokenizer')
    
    group.add_argument('--add_bos_token', action='store_true',
                       help='Append an <bos> token to the begin of a document.')
    group.add_argument('--add_eos_token', action='store_true',
                       help='Append an <eos> token to the end of a document.')
    group.add_argument('--sp_model_file', type=str, default="/home/deepspeed/projects/PLLM/Chinese-LLaMA-Alpaca/scripts/chinese_sp.model",
                       help='sentencepeice tokenizer model.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_file', type=str, default="data/test_ids.npy",
                       help='Path to output file')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=10,
                       help='Number of worker processes to launch')
    group.add_argument('--random_seed', type=int, default=1234,
                       help='Random seed for data shuffling')

    group.add_argument('--token_per_file', type=int, default=1e9,
                       help='Token number per file')
    group.add_argument('--prefix', type=str, default='wiki_zh',
                       help='Prefix of the saved file')
    args = parser.parse_args()

    # some default/dummy values for the tokenizer
    # args.rank = 0
    # args.make_vocab_size_divisible_by = 128
    # args.tensor_model_parallel_size = 1
    # args.vocab_extra_ids = 0

    return args
    

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # load
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(self.args.sp_model_file)
        # Use Encoder class as a container for global data
        Encoder.tokenizer = sp_model

    def encode_json(self, json_line):
        data = json.loads(json_line)[self.args.json_key]
        doc_ids = Encoder.tokenizer.tokenize(data) 
        if len(doc_ids) > 0 and self.args.add_bos_token:
            doc_ids.append(Encoder.tokenizer.eod)
        if self.args.add_bos_token:
            return [Encoder.tokenizer.bos] + doc_ids
        else:
            return doc_ids

    def encode_text(self, text_line):
        """
        A text_line store a whole document. 
        """
        doc_ids = Encoder.tokenizer.tokenize(text_line) 
        if len(doc_ids) > 0 and self.args.add_bos_token:
            doc_ids.append(Encoder.tokenizer.eod)
        if self.args.add_bos_token:
            return [Encoder.tokenizer.bos] + doc_ids
        else:
            return doc_ids


def process_file():
    """ Parse 单一文档
    """
    args = get_args()
    encoder = Encoder(args)
    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.map(encoder.encode_text, fin)
    idxes = np.arange(len(encoded_docs))
    np.random.seed = args.random_seed
    np.random.shuffle(idxes)
    encoded_docs = np.array(encoded_docs, dtype='object')[idxes]
    ids = np.concatenate(encoded_docs).astype('int16')
    np.save(args.output_file, ids)


def process_folder():
    """ 参考CLUENews的Wiki数据集
    1. wiki/目录下有很多嵌套子目录，子目录下存的文件都是可以用来训练模型的。
    """
    args = get_args()
    encoder = Encoder(args)
    print("Opening", args.input)
    # get all files under the foler
    assert os.path.isdir(args.input), f"Expected a folder: {args.input}"
    folders = [args.input]
    files = []
    from os import listdir
    from os.path import isfile
    while len(folders):
        folder = folders.pop()
        folder_files = listdir(folder)
        for item in folder_files:
            item = os.path.join(folder, item)
            if isfile(item):
                files.append(item)
            else:
                folders.append(item)
    # tokenize the data and save tokens to files with each file contains a given number of tokens
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    all_encoded_docs = []
    token_num = 0
    print("Find {} files...".format(len(files)))
    np.random.seed = args.random_seed
    np.random.shuffle(files)
    slice_num = 0
    for file in files:
        fin = open(file, 'r', encoding='utf-8')
        encoded_docs = pool.imap_unordered(encoder.encode_json, fin)

        for encoded_doc in encoded_docs:
            token_num += len(encoded_doc)
            all_encoded_docs.append(encoded_doc)
            if token_num >= args.token_per_file:
                save_path = os.path.join(args.output_file, 
                                '{}_slice_{}.npy'.format(args.prefix, slice_num))
                np.random.seed = args.random_seed
                np.random.shuffle(all_encoded_docs)
                ids = np.concatenate(all_encoded_docs).astype('int16')
                np.save(save_path, ids)
                slice_num += 1
                all_encoded_docs = []
                token_num = 0
    if len(all_encoded_docs) > 0:
        save_path = os.path.join(args.output_file, 
                                '{}_slice_{}.npy'.format(args.prefix, slice_num))
        np.random.seed = args.random_seed
        np.random.shuffle(all_encoded_docs)
        ids = np.concatenate(all_encoded_docs).astype('int16')
        np.save(save_path, ids)
    

if __name__ == '__main__':
    process_folder()