"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1280)
    parser.add_argument('--model', type=str, default='ban')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=8)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--logits', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=12)
    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(max_value=N)
    for v, b, q, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        logits, att = model(v, b, q, None)
        pred[idx:idx+batch_size,:].copy_(logits.data)
        qIds[idx:idx+batch_size].copy_(i)
        idx += batch_size
        if args.debug:
            print(get_question(q.data[0], dataloader))
            print(get_answer(logits.data[0], dataloader))
    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = qIds[i].item()
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset(args.split, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma).cuda()
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    def process(args, model, eval_loader):
        model_path = args.input+'/model%s.pth' % \
            ('' if 0 > args.epoch else '_epoch%d' % args.epoch)
    
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = nn.DataParallel(model).cuda()
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)

        logits, qIds = get_logits(model, eval_loader)
        results = make_json(logits, qIds, eval_loader)
        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)

        if args.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % args.index)
        
        utils.create_dir(args.output)
        if 0 <= args.epoch:
            model_label += '_epoch%d' % args.epoch

        with open(args.output+'/%s_%s.json' \
            % (args.split, model_label), 'w') as f:
            json.dump(results, f)

    process(args, model, eval_loader)
