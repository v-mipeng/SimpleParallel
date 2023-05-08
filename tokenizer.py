from logging import getLogger
from typing import List

import sentencepiece as spm
import jieba

logger = getLogger()


class JiebaTokenizer(object):

    def __init__(self, model_file, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        self.pad_id = self.sp.pad_id()

    @property
    def vocab_size(self):
        return self.sp.vocab_size()

    def __len__(self):
        return self.sp.vocab_size()

    @property
    def eod(self):
        return self.eod_id
    
    def bos_id(self):
        return self.sp.bos_id()

    def eos_id(self):
        return self.sp.eos_id()

    def eod_id(self):
        return self.sp.eod_id()

    def tokenize(self, text):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text)]
        new_seg = " ".join(seg_list)
        return self.sp.encode(new_seg)

    def convert_tokens_to_ids(self, tokens):
        return tokens

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def encode(self, text):
        res = self.tokenize(text)
        return res

    def decode(self, tokens):
        text = self.sp.decode(tokens)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text
