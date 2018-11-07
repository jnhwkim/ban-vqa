"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, Flickr30kFeatureDataset
import base_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    if args.task == 'vqa':
        from train import evaluate
        dict_path = 'data/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        eval_dset = VQAFeatureDataset('val', dictionary, adaptive=True)

    elif args.task == 'flickr':
        from train_flickr import evaluate
        dict_path = 'data/flickr30k/dictionary.pkl'
        dictionary = Dictionary.load_from_file(dict_path)
        eval_dset = Flickr30kFeatureDataset('test', dictionary)
        args.op = ''
        args.gamma = 1

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma, args.task).cuda()
    model_data = torch.load(args.input+'/model'+('_epoch%d' % args.epoch if 0 < args.epoch else '')+'.pth')

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))

    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    model.train(False)

    eval_score, bound, entropy = evaluate(model, eval_loader)
    if args.task == 'vqa':
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    elif args.task == 'flickr':
        print('\teval score: %.2f/%.2f/%.2f (%.2f)' % (
        100 * eval_score[0], 100 * eval_score[1], 100 * eval_score[2], 100 * bound))


