"""Train a custom tokenizer using sentencepiece (I prefere BPE)
"""
import os
import multiprocessing


import json
import jieba
translator = str.maketrans(" \n", "\u2582\u2583")
  

def cut(line):
    text = json.loads(line)['text']
    seg_list = [x.translate(translator) for x in jieba.cut(text)]
    text = " ".join(seg_list)
    return text

    
def cluter_file_within_folder(folder_path, output_path, workers=20):
    """ 参考CLUENews的Wiki数据集 （解压后目录）
    对于中文数据，我们参考CMP的做法，先用jieba进行分词，然后得到bpe词表。
    @Args
        folder_path (str): 目录下有很多嵌套子目录，子目录下存的文件都是可以用来训练模型的，每行对应一个dictionary，域text为训练文本
        output_path (str): 输出文件目录
        workers (int): 多进程数目
    @Return:
        None
    """
    # get all files under the foler
    from os import listdir
    from os.path import isfile
    assert os.path.isdir(folder_path), f"Expected a folder: {folder_path}"
    folders = [folder_path]
    files = []
    while len(folders):
        folder = folders.pop()
        folder_files = listdir(folder)
        for item in folder_files:
            item = os.path.join(folder, item)
            if isfile(item):
                files.append(item)
            else:
                folders.append(item)
    
    pool = multiprocessing.Pool(workers)

    with open(output_path, 'w+') as fout:
        for file in files:
            fin = open(file, 'r', encoding='utf-8')
            cut_texts = pool.imap_unordered(cut, fin)
            for text in cut_texts:
                fout.write(text+'\n')


def train_tokenizer(data_path, output_path):
    import sentencepiece as spm
    spm.SentencePieceTrainer.train('--input={} --model_prefix={} --vocab_size=20000 --model_type=bpe --character_coverage=0.9995 --max_sentence_length=10000'.format(data_path, output_path))
    

if __name__ == '__main__':
    cluter_file_within_folder('/home/deepspeed/projects/PLLM/data/CLUECorpusSmall/wiki_zh/', '/home/deepspeed/projects/PLLM/data/wiki_zh_full.txt')
    train_tokenizer('/home/deepspeed/projects/PLLM/data/wiki_zh_full.txt', '/home/deepspeed/projects/PLLM/data/wiki_zh_bpe')

