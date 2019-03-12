"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import utils
from attention import BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter


class BanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))

        return logits, att

class BanModel_flickr(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, op, glimpse):
        super(BanModel_flickr, self).__init__()
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.alpha = torch.Tensor([1.]*(glimpse))

    # features, spatials, sentence, e_pos, target
    def forward(self, v, b, q, e, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch, seq_length]
        e: [batch, num_entities]

        return: logits, not probs
        """
        assert q.size(1) > e.data.max(), 'len(q)=%d > e_pos.max()=%d' % (q.size(1), e.data.max())
        MINUS_INFINITE = -99
        if 's' in self.op:
            v = torch.cat([v, b], 2)
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        # entity positions
        q_emb = utils.batched_index_select(q_emb, 1, e)

        att = self.v_att.forward_all(v, q_emb, True, True, MINUS_INFINITE)  # b x g x v x q
        mask = (e == 0).unsqueeze(1).unsqueeze(2).expand(att.size())
        mask[:, :, :, 0].data.fill_(0)  # at least one entity per sentence
        att.data.masked_fill_(mask.data, MINUS_INFINITE)

        return None, att


def build_ban(dataset, num_hid, op='', gamma=4, task='vqa'):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    v_att = BiAttention(dataset.v_dim, num_hid, num_hid, gamma)
    if task == 'vqa':
        b_net = []
        q_prj = []
        c_prj = []
        objects = 10  # minimum number of boxes
        for i in range(gamma):
            b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
            q_prj.append(FCNet([num_hid, num_hid], '', .2))
            c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
        classifier = SimpleClassifier(
            num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
        counter = Counter(objects)
        return BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, gamma)
    elif task == 'flickr':
        return BanModel_flickr(w_emb, q_emb, v_att, op, gamma)
