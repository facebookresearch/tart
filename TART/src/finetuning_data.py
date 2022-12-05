# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
import sys
import numpy as np
from src import normalize_text


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        negative_ctxs=1,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        hard_order=False
    ):
        self.negative_ctxs = negative_ctxs
        self.hard_negative_min_idx = negative_hard_min_idx
        self.negative_hard_ratio = negative_hard_ratio
        self.negative_hard_min_idx = negative_hard_min_idx
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        self.hard_order = hard_order

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        if self.training:
            gold = random.choice(example["positive_ctxs"])
            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []
            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.hard_negative_min_idx :], n_hard_negatives
                )
                negatives += hard_negatives
        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []
        
        if "title" in gold and type(gold["title"]) is str and len(gold) > 0:
            gold = gold["title"] + " " + gold["text"]
        else:
            gold = gold["text"]

        negatives_new = []
        for n in negatives:
            if "title" in n and type(n["title"]) is str and len(n["title"]) > 0:
                negatives_new.append(n["title"] + " " + n["text"])
            else:
                negatives_new.append(n["text"])
        
        negatives = negatives_new
        # negatives = [
        #     n["title"] + " " + n["text"] if ("title" in n and type(n["title"]) is str and len(n["title"]) > 0) else n["text"] for n in negatives
        # ]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives],
        }
        return example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def sample_n_hard_negatives(self, ex):
        
        if "hard_negative_ctxs" in ex:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0
        n_random_negatives = self.negative_ctxs - n_hard_negatives
        if "negative_ctxs" in ex:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        # print(n_hard_negatives, n_random_negatives)
        return n_hard_negatives, n_random_negatives


class Collator(object):
    def __init__(self, tokenizer, passage_maxlength=200):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        golds = [ex["gold"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        n_tokens, n_mask = k_tokens[len(golds) :], k_mask[len(golds) :]

        batch = {
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        return batch

class DatasetKD(torch.utils.data.Dataset):
    def __init__(self,
                 datapaths,
                 n_context=50,
                 question_prefix='',
                 title_prefix='',
                 passage_prefix='',
                 global_rank=-1,
                 world_size=-1,
                 maxload=None, 
                 random_sort=False):

        self._load_data(datapaths, global_rank, world_size, maxload)
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.random_sort = random_sort
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            if len(example['ctxs']) > self.n_context and self.random_sort is True:
                contexts = random.sample(example['ctxs'], k=self.n_context)
            else:
                contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['gold_score']) for c in contexts]
            scores = torch.tensor(scores)
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            'index' : index,
            'question' : question,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'gold_score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['gold_score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

class CollatorKD(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        batch = {
            "question_ids": question_ids,
            "question_mask": question_mask,
            "passage_ids": passage_ids,
            "passage_mask": passage_masks,
            "gold_score": scores
        }

        return batch
