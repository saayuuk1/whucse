from os.path import join
from codecs import open
import json
from models.config import TrainingConfig
import numpy as np
from tqdm import tqdm
import torch

def build_corpus(split, make_vocab = True, data_dir="./ResumeNER"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    with open(TrainingConfig.pretrained_emb, 'r', encoding='utf-8') as f:
        word2vec_dict = json.loads(f.read())
    with open(join(data_dir, "word2id.json"), 'r', encoding='utf-8') as f:
        word2id = json.loads(f.read())
    tag2id = build_map(tag_lists)

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word_embedding_init_matrix = np.random.normal(-1, 1, size=(len(word2id), TrainingConfig.emb_size))
        count_in = 0
        for word, idx in tqdm(word2id.items(), desc="Embedding matrix initializing..."):
            if word in word2vec_dict.keys():
                count_in += 1
                word_embedding_init_matrix[idx] = word2vec_dict[word]

        print("{:.4f} tokens are in the pretrain word embedding matrix".format(count_in / len(word2id)))  # 命中预训练词向量的比例
        word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

        return word_lists, tag_lists, word2id, tag2id, word_embedding_init_matrix
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
