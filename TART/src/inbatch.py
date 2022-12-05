# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import logging
import torch.distributed as dist
import copy
from src import contriever, dist_utils, utils
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever

    def _load_retriever(self, model_id, pooling, random_init):
        print("load retrieval")
        print(model_id)
        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        elif "t5" in model_id or "T0" in model_id or "gtr" in model_id:
            print("loading t0")
            model_class = contriever.T5Contriever
            print(model_class)
        else:
            model_class = contriever.Contriever

        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)
        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, gold_scores=None, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats



class ByInBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(ByInBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.q_encoder = copy.deepcopy(retriever)
        self.p_encoder = copy.deepcopy(retriever)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_q_encoder(self):
        return self.q_encoder

    def get_p_encoder(self):
        return self.p_encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.q_encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.p_encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats


class InBatch_KD(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None, loss_type="kl", temperature=1):
        super(InBatch_KD, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.loss_type = loss_type
        self.temperature = temperature
        if loss_type == "kl":
            self.loss_fct = torch.nn.KLDivLoss()
        elif loss_type == "mse":
            self.loss_fct = torch.nn.MSELoss()
        else:
            raise NotImplementedError

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        elif "t5" in model_id or "T0" in model_id or "gtr" in model_id:
            model_class = contriever.T5Contriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, question_ids, question_mask, passage_ids, passage_mask, gold_score, stats_prefix="", iter_stats={}, **kwargs):
        question_output = self.encoder(input_ids=question_ids, attention_mask=question_mask, normalize=self.norm_query)
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.encoder(input_ids=passage_ids, attention_mask=passage_mask, normalize=self.norm_doc)

        score = torch.einsum(
            'bd,bid->bi',
            question_output,
            passage_output.view(bsz, n_passages, -1)
        )

        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            if self.loss_type == "kl":
                loss = self.kldivloss(score, gold_score) 
            else:
                loss = self.mseloss(score, gold_score) 
        else:
            loss = None
        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        # predicted_idx = torch.argmax(scores, dim=-1)
        # accuracy = 100 * (predicted_idx == labels).float().mean()
        # stdq = torch.std(qemb, dim=0).mean().item()
        # stdk = torch.std(kemb, dim=0).mean().item()
        # iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        # iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        # iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)
        # print(loss)
        return loss, iter_stats

    def kldivloss(self, score, gold_score):
        # print("scores")
        # print(gold_score[0,:10])
        gold_score = torch.softmax(gold_score / self.temperature, dim=-1)
        # print(gold_score[0,:10])
        # print(score[0,:10])
        score = torch.nn.functional.log_softmax(score / self.temperature, dim=-1) 
        # print(score[0,:10])
        loss = self.loss_fct(score, gold_score)  * (self.temperature**2) 
        # loss = F.kl_div(score, gold_score, size_average=False) * (self.temperature**2) 
        # print(loss)
        # print(loss.size())

        return loss

    def mseloss(self, score, gold_score):
        # print("scores")
        # print(gold_score[0,:10])
        gold_score = torch.softmax(gold_score, dim=-1)
        # print(gold_score[0,:10])
        # print(score[0,:10])
        score = torch.softmax(score, dim=-1)
        # print(score[0,:10])
        loss = self.loss_fct(score, gold_score)
        # print(loss)
        # print(loss.size())
        return self.loss_fct(score, gold_score)