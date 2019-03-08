"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import argparse
import sys
import json
import _pickle as cPickle
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary
from utils import get_sent_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    args = parser.parse_args()
    return args



def create_dictionary(dataroot, task='vqa'):
    dictionary = Dictionary()
    if task == 'vqa':
        files = [
            'v2_OpenEnded_mscoco_train2014_questions.json',
            'v2_OpenEnded_mscoco_val2014_questions.json',
            'v2_OpenEnded_mscoco_test2015_questions.json',
            'v2_OpenEnded_mscoco_test-dev2015_questions.json'
        ]
        for path in files:
            question_path = os.path.join(dataroot, path)
            qs = json.load(open(question_path))['questions']
            for q in qs:
                dictionary.tokenize(q['question'], True)

    elif task == 'flickr':
        files = [
            'train_ids.pkl',
            'val_ids.pkl',
            'test_ids.pkl',
        ]
        sentence_dir = os.path.join(dataroot, 'Flickr30kEntities/Sentences')

        for path in files:
            ids_file = os.path.join(dataroot, path)

            with open(ids_file, 'rb') as f:
                imgids = cPickle.load(f)

            for image_id in imgids:
                question_path = os.path.join(sentence_dir, '%d.txt' % image_id)
                phrases = get_sent_data(question_path)
                for phrase in phrases:
                    dictionary.tokenize(phrase, True)
    return dictionary


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
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    args = parse_args()
    dataroot = 'data' if args.task == 'vqa' else 'data/flickr30k'

    dictionary_path = os.path.join(dataroot, 'dictionary.pkl')

    d = create_dictionary(dataroot, args.task)
    d.dump_to_file(dictionary_path)

    d = Dictionary.load_from_file(dictionary_path)
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(dataroot, 'glove6b_init_%dd.npy' % emb_dim), weights)
