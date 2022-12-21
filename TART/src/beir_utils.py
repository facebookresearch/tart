# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict
from typing import List, Dict
import numpy as np
import torch
import torch.distributed as dist
import json

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch

from beir.reranking.models import CrossEncoder
from src.rerank import Rerank
from tqdm import tqdm
import glob
import src.dist_utils as dist_utils
from src import normalize_text
import jsonlines
import pandas as pd
import csv

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        prompt=None,
        emb_load_path=None,
        emb_save_path=None,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.prompt = prompt
        self.emb_load_path = emb_load_path
        self.emb_save_path = emb_save_path

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.prompt is not None:
            queries = ["{0} [SEP] {1}".format(self.prompt, query) for query in queries]
            print(queries[-1])

        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        
        if self.emb_load_path is None:
            corpus = [corpus[i] for i in idx]
            corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
            # corpus = ["question" + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
            if self.normalize_text:
                corpus = [normalize_text.normalize(c) for c in corpus]
            if self.lower_case:
                corpus = [c.lower() for c in corpus]

            allemb = []
            nbatch = (len(corpus) - 1) // batch_size + 1
            with torch.no_grad():
                for k in tqdm(range(nbatch)):
                    start_idx = k * batch_size
                    end_idx = min((k + 1) * batch_size, len(corpus))

                    cencode = self.tokenizer.batch_encode_plus(
                        corpus[start_idx:end_idx],
                        max_length=self.max_length,
                        padding=True,
                        truncation=True,
                        add_special_tokens=self.add_special_tokens,
                        return_tensors="pt",
                    )
                    cencode = {key: value.cuda() for key, value in cencode.items()}
                    emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                    allemb.append(emb.cpu())

            allemb = torch.cat(allemb, dim=0)
            if self.emb_save_path is not None:
                torch.save(allemb, self.emb_save_path)
        else:
            print("loading from {}".format(self.emb_load_path))
            embs_list = []
            for path in self.emb_load_path:
                embs = torch.load(path)
                print(embs.size())
                embs_list.append(embs)
                print(len(embs_list))
                
            allemb = torch.cat(embs_list, dim=0)
            print(allemb.size())
            
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb


def evaluate_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    prompt=None,
    emb_load_path=None,
    emb_save_path=None,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
            prompt=prompt,
            emb_load_path=emb_load_path,
            emb_save_path=emb_save_path,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        results = retriever.retrieve(corpus, queries)
        if is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(results, f"{save_results_path}")
    elif dataset == "cqadupstack":  # compute macroaverage over datasets
        paths = glob.glob(data_path)
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            results = retriever.retrieve(corpus, queries)
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics

def evaluate_model_multiple(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    prompt=None,
    multiple_prompts=None
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    global_results = {}
    for prompt in tqdm(multiple_prompts):
        dmodel = DenseRetrievalExactSearch(
            DenseEncoderModel(
                query_encoder=query_encoder,
                doc_encoder=doc_encoder,
                tokenizer=tokenizer,
                add_special_tokens=add_special_tokens,
                norm_query=norm_query,
                norm_doc=norm_doc,
                lower_case=lower_case,
                normalize_text=normalize_text,
                prompt=prompt
            ),
            batch_size=batch_size,
        )
        retriever = EvaluateRetrieval(dmodel, score_function=score_function)
        data_path = os.path.join(beir_dir, dataset)

        if not os.path.isdir(data_path) and is_main:
            url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
            data_path = beir.util.download_and_unzip(url, beir_dir)
        dist_utils.barrier()

        if not dataset == "cqadupstack":
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            results = retriever.retrieve(corpus, queries)
            for q_id, result in results.items():
                global_results.setdefault(q_id, {})
                for doc_id, score in result.items():
                    global_results[q_id].setdefault(doc_id, 0.0)
                    global_results[q_id][doc_id] += score
        elif dataset == "cqadupstack":  # compute macroaverage over datasets
            paths = glob.glob(data_path)
            for path in paths:
                corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
                results = retriever.retrieve(corpus, queries)
            for q_id, result in results.items():
                global_results.setdefault(q_id, {})
                for doc_id, score in result.items():
                    global_results[q_id].setdefault(doc_id, 0.0)
                    global_results[q_id][doc_id] += score
    
    results = global_results

    if is_main:
        if not dataset == "cqadupstack":
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(results, f"{save_results_path}")

        elif dataset == "cqadupstack": 
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            for key, value in metrics.items():
                assert (
                    len(value) == 12
                ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics

def evaluate_model_ce(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    prompt=None,
    ce_prompt=None,
    ce_model_path=None,
    load_retrieval_results=False
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
            prompt=prompt,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = os.path.join(beir_dir, dataset)
    # cross_encoder_model = CrossEncoder(model_path='/checkpoint/akariasai/ranker/bert_base_st_ranker_manual_all_with_instructions_hard_negatives_instructions', num_labels=2)
    # reranker = Rerank('/checkpoint/akariasai/ranker/bert_base_st_ranker_manual_all_with_instructions_hard_negatives_instructions_instruction_unfollowing/checkpoint-50000/', batch_size=100)
    reranker = Rerank(ce_model_path, batch_size=100)
    
    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    dist_utils.barrier()

    if not dataset == "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        if load_retrieval_results is True:
            results = json.load(open("retriever_results_contriever_{}.json".format(dataset)))
        else:
            results = retriever.retrieve(corpus, queries)
            with open("retriever_results_contriever_{}.json".format(dataset), "w") as outfile:
                json.dump(results, outfile)

        print("start reranking")
        rerank_results = reranker.rerank(corpus, queries, results, top_k=100, prompt=ce_prompt)

        if is_main:
            ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, retriever.k_values)
            for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                if isinstance(metric, str):
                    metric = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric=metric)
                for key, value in metric.items():
                    metrics[key].append(value)
            if save_results_path is not None:
                torch.save(rerank_results, f"{save_results_path}")

    elif dataset == "cqadupstack":  # compute macroaverage over datasetds
        paths = glob.glob(data_path)
        for path in paths:
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
            results = retriever.retrieve(corpus, queries)
            rerank_results = reranker.rerank(corpus, queries, results, top_k=100, prompt=ce_prompt)
            if is_main:
                ndcg, _map, recall, precision = retriever.evaluate(qrels, rerank_results, retriever.k_values)
                for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
                    if isinstance(metric, str):
                        metric = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric=metric)
                    for key, value in metric.items():
                        metrics[key].append(value)
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics

def evaluate_lotte_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    qa_file,
    corpus_file, 
    cat_file, 
    batch_size=128,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    save_results_path=None,
    lower_case=False,
    normalize_text=False,
    prompt=None,
    emb_load_path=None,
    emb_save_path=None,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
            emb_load_path=emb_load_path,
            emb_save_path=emb_save_path,
        ),
        batch_size=batch_size,
    )
    retriever = EvaluateRetrieval(dmodel, score_function=score_function)
    data_path = process_lotte_data(qa_file, corpus_file, cat_file, prompt)

    dist_utils.barrier()

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    results = retriever.retrieve(corpus, queries)

    if is_main:
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
        for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
            if isinstance(metric, str):
                metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
            for key, value in metric.items():
                metrics[key].append(value)
        if save_results_path is not None:
            torch.save(results, f"{save_results_path}")
        for key, value in metrics.items():
            assert (
                len(value) == 12
            ), f"cqadupstack includes 12 datasets, only {len(value)} values were compute for the {key} metric"

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics

def process_lotte_data(qa_file, corpus_file, cat_file, prompt=None):
    formatted_corpus_data = []
    corpus_data = pd.read_csv(corpus_file, sep="\t")
    output_dir_name = "lotte_search"
    print("reformatting data")
    if os.path.exists(output_dir_name) is False:
        os.mkdir(output_dir_name)
    print("loading qid2cat")
    qid2category = json.load(open(cat_file))
    print(len(qid2category))
    for idx, item in corpus_data.iterrows():
        doc_id = item[0]
        text = item[1]
        formatted_corpus_data.append({"_id": str(doc_id), "text": text, "title": "", "metadata": {}})

    with jsonlines.open(os.path.join(output_dir_name, 'corpus.jsonl'), 'w') as writer:
        writer.write_all(formatted_corpus_data)
    
    qa_data = []

    with open(qa_file) as f:
        for line in f:
            qa_data.append(json.loads(line))
    
    queries = []
    qrels = []
    for item in qa_data:
        q_id = item["qid"]
        query = item["query"]
        category = qid2category[query]
        answer_pids = item["answer_pids"]
        print(prompt)
        if prompt is not None:
            prompted_query = "{0} from StackExchange {1} forum [SEP] {2}".format(prompt, category, query)
        else:
            prompted_query = query
        queries.append({"_id": str(q_id), "text":prompted_query, "metadata": {} })
        for corpus_id in answer_pids:
            qrels.append({"corpus-id": str(corpus_id), "query-id": str(q_id), "score": 1})

    if os.path.exists(os.path.join(output_dir_name, "qrels")) is False:
        os.mkdir(os.path.join(output_dir_name, "qrels"))

    with open(os.path.join(output_dir_name,  "qrels", 'test.tsv'), 'w') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['query-id', 'corpus-id', 'score'])
        for item in qrels:
            tsv_writer.writerow([item['query-id'], item['corpus-id'], item['score']])

    with jsonlines.open(os.path.join(output_dir_name, 'queries_w_instructions_sep.jsonl'), 'w') as writer:
        writer.write_all(queries)

    print(queries[-1])
    print(qa_data[-1])
    print(formatted_corpus_data[-1])

    return output_dir_name