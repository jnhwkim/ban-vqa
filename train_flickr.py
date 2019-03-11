"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""

import os
import time
import torch
import torch.nn as nn
import utils
from train import instance_bce_with_logits, compute_score_with_logits


def compute_recall_with_logits(logits, labels):
    logits = torch.sort(logits, 1, descending=True)[1].data
    scores = [0]*3
    for i,r in enumerate([1,5,10]):
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits[:,:r], 1)
        scores[i] = ((one_hots * labels).sum(1)>=1).float().sum()
    return scores


def train(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    lr_default = 1e-3 if eval_loader is not None else 7e-4
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = 0

        if epoch < len(gradual_warmup_steps):
            optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            optim.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % optim.param_groups[0]['lr'])

        for i, (v, b, p, e, n, a, idx, types) in enumerate(train_loader):
            v = v.cuda()
            b = b.cuda()
            p = p.cuda()
            e = e.cuda()
            a = a.cuda()

            _, logits = model(v, b, p, e, a)
            n_obj = logits.size(2)
            logits.squeeze_()

            merged_logit = torch.cat(tuple(logits[j, :, :n[j][0]] for j in range(n.size(0))), -1).permute(1, 0)
            merged_a = torch.cat(tuple(a[j, :n[j][0], :n_obj] for j in range(n.size(0))), 0)

            loss = instance_bce_with_logits(merged_logit, merged_a, 'sum') / v.size(0)
            N += n.sum().float()

            batch_score = compute_score_with_logits(merged_logit, merged_a.data).sum()

            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()
            total_loss += loss.item() * v.size(0)
            train_score += batch_score.item()

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score, bound, entropy = evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f/%.2f/%.2f (%.2f)' % (
            100 * eval_score[0], 100 * eval_score[1], 100 * eval_score[2], 100 * bound))
            eval_score = eval_score[0]

        if eval_loader is not None and entropy is not None:
            info = ''
            for i in range(entropy.size(0)):
                info = info + ' %.2f' % entropy[i]
            logger.write('\tentropy: ' + info)

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch >= saving_epoch):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score


@torch.no_grad()
def evaluate(model, dataloader):
    upper_bound = 0
    entropy = None
    score = [0] * 3
    N = 0
    for i, (v, b, p, e, n, a, idx, types) in enumerate(dataloader):
        v = v.cuda()
        b = b.cuda()
        p = p.cuda()
        e = e.cuda()
        a = a.cuda()
        _, logits = model(v, b, p, e, None)
        n_obj = logits.size(2)
        logits.squeeze_()

        merged_logits = torch.cat(tuple(logits[j, :, :n[j][0]] for j in range(n.size(0))), -1).permute(1, 0)
        merged_a = torch.cat(tuple(a[j, :n[j][0], :n_obj] for j in range(n.size(0))), 0)

        recall = compute_recall_with_logits(merged_logits, merged_a.data)
        for r_idx, r in enumerate(recall):
            score[r_idx] += r
        N += n.sum().float()
        upper_bound += merged_a.max(-1, False)[0].sum().item()

    for i in range(3):
        score[i] = score[i] / N
    upper_bound = upper_bound / N

    return score, upper_bound, entropy
