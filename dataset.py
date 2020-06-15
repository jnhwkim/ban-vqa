"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
from xml.etree.ElementTree import parse
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools
import re

COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
        (name + '2014' if 'test'!=name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if 'test'!=name[:4]: # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id], question, answer))
    else: # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id], question, None))

    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot, 'question_answers.json')
    image_data_path = os.path.join(dataroot, 'image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' % (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = cPickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = cPickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        _vgv = json.load(open(image_data_path, 'r')) #108,077
        vgv = {}
        for _v in _vgv: 
            if None != _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        counts = [0, 0, 0, 0] # used image, used question, total question, out-of-split
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if None != coco_id:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if None == img_idx:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(q['answer'])
                    label = ans2label.get(_answer, None)
                    if None != label and None != img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id' : q['id'],
                            'image_id'    : coco_id,
                            'image'       : img_idx,
                            'question'    : q['question'],
                            'answer'      : answer}
                        if not COUNTING_ONLY or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' % \
            (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' % \
            (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' % \
            (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            cPickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['id']==vgv_id:
            return v['coco_id']
    return None


def _load_flickr30k(dataroot, img_id2idx, bbox, pos_boxes):
    """Load entries

    img_id2idx: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    pattern_phrase = r'\[(.*?)\]'
    pattern_no = r'\/EN\#(\d+)'

    missing_entity_count = dict()
    multibox_entity_count = 0

    entries = []
    for image_id, idx in img_id2idx.items():

        phrase_file = os.path.join(dataroot, 'Flickr30kEntities/Sentences/%d.txt' % image_id)
        anno_file = os.path.join(dataroot, 'Flickr30kEntities/Annotations/%d.xml' % image_id)

        with open(phrase_file, 'r', encoding='utf-8') as f:
            sents = [x.strip() for x in f]

        # Parse Annotation
        root = parse(anno_file).getroot()
        obj_elems = root.findall('./object')
        pos_box = pos_boxes[idx]
        bboxes = bbox[pos_box[0]:pos_box[1]]
        target_bboxes = {}

        for elem in obj_elems:
            if elem.find('bndbox') == None or len(elem.find('bndbox')) == 0:
                continue
            left = int(elem.findtext('./bndbox/xmin'))
            top = int(elem.findtext('./bndbox/ymin'))
            right = int(elem.findtext('./bndbox/xmax'))
            bottom = int(elem.findtext('./bndbox/ymax'))
            assert 0 < left and 0 < top

            for name in elem.findall('name'):
                entity_id = int(name.text)
                assert 0 < entity_id
                if not entity_id in target_bboxes:
                    target_bboxes[entity_id] = []
                else:
                    multibox_entity_count += 1
                target_bboxes[entity_id].append([left, top, right, bottom])

        # Parse Sentence
        for sent_id, sent in enumerate(sents):
            sentence = utils.remove_annotations(sent)
            entities = re.findall(pattern_phrase, sent)
            entity_indices = []
            target_indices = []
            entity_ids = []
            entity_types = []

            for entity_i, entity in enumerate(entities):
                info, phrase = entity.split(' ', 1)
                entity_id = int(re.findall(pattern_no, info)[0])
                entity_type = info.split('/')[2:]

                entity_idx = utils.find_sublist(sentence.split(' '), phrase.split(' '))
                assert 0 <= entity_idx

                if not entity_id in target_bboxes:
                    if entity_id >= 0:
                        missing_entity_count[entity_type[0]] = missing_entity_count.get(entity_type[0], 0) + 1
                    continue

                assert 0 < entity_id

                entity_ids.append(entity_id)
                entity_types.append(entity_type)

                target_idx = utils.get_match_index(target_bboxes[entity_id], bboxes)
                entity_indices.append(entity_idx)
                target_indices.append(target_idx)

            if 0 == len(entity_ids):
                continue

            entries.append(
                _create_flickr_entry(idx, sentence, entity_indices, target_indices, entity_ids, entity_types))

    if 0 < len(missing_entity_count.keys()):
        print('missing_entity_count=')
        print(missing_entity_count)
        print('multibox_entity_count=%d' % multibox_entity_count)

    return entries


# idx, sentence, entity_indices, target_indices, entity_ids, entity_types
def _create_flickr_entry(img, sentence, entity_indices, target_indices, entity_ids, entity_types):
    type_map = {'people':0,'clothing':1,'bodyparts':2,'animals':3,'vehicles':4,'instruments':5,'scene':6,'other':7}
    MAX_TYPE_NUM = 3
    for i, entity_type in enumerate(entity_types):
        assert MAX_TYPE_NUM >= len(entity_type)
        entity_types[i] = list(type_map[x] for x in entity_type)
        entity_types[i] += [-1] * (MAX_TYPE_NUM-len(entity_type))
    entry = {
        'image'          : img,
        'sentence'       : sentence,
        'entity_indices' : entity_indices,
        'target_indices' : target_indices,
        'entity_ids'     : entity_ids,
        'entity_types'   : entity_types,
        'entity_num'     : len(entity_ids)}
    return entry



class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        
        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        print('loading features from h5 file')    
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]
            spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target
        else:
            return features, spatials, question, question_id

    def __len__(self):
        return len(self.entries)


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, features, spatials, dictionary, dataroot='data', adaptive=False, pos_boxes=None):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
                open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        self.features = features
        self.spatials = spatials
        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]
            spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1],:]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, spatials, question, target

    def __len__(self):
        return len(self.entries)


class Flickr30kFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data/flickr30k/'):
        super(Flickr30kFeatureDataset, self).__init__()

        self.num_ans_candidates = 100

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s_imgid2idx.pkl' % name), 'rb'))

        h5_path = os.path.join(dataroot, '%s.hdf5' % name)

        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            self.bbox = np.array(hf.get('image_bb'))
            self.pos_boxes = np.array(hf.get('pos_boxes'))

        self.entries = _load_flickr30k(dataroot, self.img_id2idx, self.bbox, self.pos_boxes)
        self.tokenize()
        self.tensorize(self.num_ans_candidates)
        self.v_dim = self.features.size(1)
        self.s_dim = self.spatials.size(1)

    def tokenize(self, max_length=82):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['sentence'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['p_token'] = tokens

    def tensorize(self, max_box=100, max_entities=16, max_length=82):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            phrase = torch.from_numpy(np.array(entry['p_token']))
            entry['p_token'] = phrase

            assert len(entry['target_indices']) == entry['entity_num']
            assert len(entry['entity_indices']) == entry['entity_num']

            target_tensors = []
            for i in range(entry['entity_num']):
                target_tensor = torch.zeros(1, max_box)
                if len(entry['target_indices'][i]) > 0:
                    target_idx = torch.from_numpy(np.array(entry['target_indices'][i]))
                    target_tensor = torch.zeros(max_box).scatter_(0, target_idx, 1).unsqueeze(0)
                target_tensors.append(target_tensor)
            assert len(target_tensors) <= max_entities, '> %d entities!' % max_entities
            for i in range(max_entities - len(target_tensors)):
                target_tensor = torch.zeros(1, max_box)
                target_tensors.append(target_tensor)
                entry['entity_ids'].append(0)
            # padding entity_indices with non-overlapping indices
            entry['entity_indices'] += [x for x in range(max_length) if x not in entry['entity_indices']]
            entry['entity_indices'] = entry['entity_indices'][:max_entities]
            entry['target'] = torch.cat(target_tensors, 0)
            # entity positions in (e) tensor
            entry['e_pos'] = torch.LongTensor(entry['entity_indices'])
            entry['e_num'] = torch.LongTensor([entry['entity_num']])
            entry['entity_ids'] = torch.LongTensor(entry['entity_ids'])
            entry['entity_types'] = torch.LongTensor(entry['entity_types'])

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        sentence = entry['p_token']
        e_pos = entry['e_pos']
        e_num = entry['e_num']
        target = entry['target']
        entity_ids = entry['entity_ids']
        entity_types = entry['entity_types']

        return features, spatials, sentence, e_pos, e_num, target, entity_ids, entity_types

    def __len__(self):
        return len(self.entries)




def tfidf_from_questions(names, dictionary, dataroot='data', target=['vqa', 'vg', 'cap', 'flickr']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'vqa' in target: # VQA 2.0
        for name in names:
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
                (name + '2014' if 'test'!=name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    if 'vg' in target: # Visual Genome
        question_path = os.path.join(dataroot, 'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    if 'cap' in target: # MSCOCO Caption
        for split in ['train2017', 'val2017']:
            captions = json.load(open('data/annotations/captions_%s.json' % split, 'r'))
            for caps in captions['annotations']:
                populate(inds, df, caps['caption'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights


if __name__=='__main__':
    dictionary = Dictionary.load_from_file('data/flickr30k/dictionary.pkl')
    tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

if __name__=='__main2__':
    from torch.utils.data import DataLoader

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('val', dictionary, adaptive=True)
    # name = 'train'
    # eval_dset = VQAFeatureDataset(name, dictionary)
    # vg_dset = VisualGenomeFeatureDataset(name, eval_dset.features, eval_dset.spatials, dictionary)

    # train_loader = DataLoader(vg_dset, 10, shuffle=True, num_workers=1)

    loader = DataLoader(train_dset, 10, shuffle=True, num_workers=1, collate_fn=utils.trim_collate)
    for i, (v, b, q, a) in enumerate(loader):
        print(v.size()) 

# VisualGenome Train
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 17464/51487 (0.3392)
#     Used VG questions: 325311/726932 (0.4475)

# VisualGenome Val
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 34023/51487 (0.6608)
#     Used VG questions: 166409/726932 (0.2289)
