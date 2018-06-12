"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import sys
import json
import functools
import operator
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
import _pickle as cPickle


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    count = 0
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            updates = 0
            for w in word.split(' '):
                if w not in word2emb:
                    continue
                weights[idx] += word2emb[w]
                updates += 1
            if updates == 0:
                count+= 1
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    emb_dims = [50, 100, 200, 300]
    weights = [0] * len(emb_dims)
    label2ans = cPickle.load(open('data/cache/trainval_label2ans.pkl', 'rb'))

    for idx, emb_dim in enumerate(emb_dims): # available embedding sizes
        glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
        weights[idx], word2emb = create_glove_embedding_init(label2ans, glove_file)
    np.save('data/glove6b_emb_%dd.npy' % functools.reduce(operator.add, emb_dims), np.hstack(weights))
