# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
import random
from pathlib import Path
import jsonlines

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import transformers

import pytrec_eval

import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
from typing import Type, List, Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, HfArgumentParser, default_data_collator
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.modeling_enc_t5 import EncT5ForSequenceClassification
from src.tokenization_enc_t5 import EncT5Tokenizer
from datasets import load_dataset

import copy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from custom_metrics import mrr, recall_cap, hole, top_k_accuracy

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    early_stopping_patience: int = field(
        default=5,
        metadata={
            "help": "Use with metric_for_best_model to stop training when the specified metric worsens "
            "for early_stopping_patience evaluation calls."
        },
    )


class Rerank:
    def __init__(self, model_name_or_path, batch_size: int = 128, **kwargs):
        if "t0" in model_name_or_path or "t5" in model_name_or_path:
            self.model = EncT5ForSequenceClassification.from_pretrained(model_name_or_path)
            self.tokenizer =  EncT5Tokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            self.tokenizer =  AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.rerank_results = {}
        self.data_collator = default_data_collator

    def preprocess_function(self, examples):
        # Tokenize the texts
        sentence1_key = "query"
        sentence2_key = "corpus"
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = self.tokenizer(*args, padding="max_length", max_length=512, truncation=True, pad_to_max_length=True)

        # # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int, 
               prompt: str = None) -> Dict[str, Dict[str, float]]:
        
        self.rerank_results = {query_id: {} for query_id in results}
        input_source_data = []
        meta_data = {}
        count = 0
        queries_dict = {item["_id"]: item["text"] for item in queries}
        for i, query_id in enumerate(tqdm(results)):

            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    if prompt is None:
                        query = queries_dict[query_id]
                    else:
                        query = "{0} [SEP] {1}".format(prompt, queries_dict[query_id])
                    input_source_data.append({"query": query, "corpus": corpus_text})
                    meta_data[count] = {"query_id": query_id, "doc_id": doc_id}
                    count += 1
            
            else:
                for doc_id in results[query_id]:
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    if prompt is None:
                        query = queries_dict[query_id]
                    else:
                        query = "{0} [SEP] {1}".format(prompt, queries_dict[query_id])
                    input_source_data.append({"query": query, "corpus": corpus_text})
                    meta_data[count] = {"query_id": query_id, "doc_id": doc_id}
                    count += 1
        
        print("setup data")
        test_file = "tmp_input_data.json"
        with jsonlines.open(test_file, 'w') as writer:
            writer.write_all(input_source_data)

        data_files = {"test": test_file}
        raw_datasets = load_dataset("json", data_files=data_files)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
        def compute_metrics(p):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        print("setup trainer")
        trainer = Trainer(
            model=self.model,
            args=None,
            train_dataset= None,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        predict_datasets = [raw_datasets["test"]]
        for predict_dataset in predict_datasets:
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predicted_scores = copy.deepcopy(predictions)
        score_preds = {}
        for index, item in enumerate(predicted_scores):
            if len(item) == 2:
                score_preds[index] = float(torch.nn.functional.softmax(torch.Tensor(item), dim=0)[1])
            else:
                score_preds[index] = float(item)
        
        reranked_results = {}
        for idx, pred in score_preds.items():
            query_id = meta_data[idx]["query_id"]
            doc_id = meta_data[idx]["doc_id"]
            reranked_results.setdefault(query_id, {})
            reranked_results[query_id][doc_id] = pred

        return reranked_results


def evaluate(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int],
                ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    if ignore_identical_ids:
        logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0
    
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)
    
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
    
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
    
    for eval in [ndcg, _map, recall, precision]:
        logging.info("\n")
        for k in eval.keys():
            logging.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision

def _load_qrels(qrels_file):
    qrels = {}
    reader = csv.reader(open(qrels_file, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)
    
    for id, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels

def evaluate_custom(qrels: Dict[str, Dict[str, int]], 
                results: Dict[str, Dict[str, float]], 
                k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
    
    if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
        return mrr(qrels, results, k_values)
    
    elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
        return recall_cap(qrels, results, k_values)
    
    elif metric.lower() in ["hole", "hole@k"]:
        return hole(qrels, results, k_values)
    
    elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
        return top_k_accuracy(qrels, results, k_values)

def embed_queries(args, queries, model, tokenizer):
    model.eval()
    embeddings, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = src.normalize_text.normalize(q)
            batch_question.append(q)

            if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=args.question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch)
                embeddings.append(output.cpu())

                batch_question = []

    embeddings = torch.cat(embeddings, dim=0)
    print(f"Questions embeddings shape: {embeddings.size()}")

    return embeddings.numpy()


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    results = {}
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        qid = d["_id"]
        results.setdefault(qid, {})
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"] if "title" in docs[c] else "",
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]
        for c in d["ctxs"]:
            results[qid][c["id"]] = float(c["score"])
    return results

# fix me
def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):

    print(f"Loading model from: {args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(args.model_name_or_path)
    model.eval()
    model = model.cuda()
    if not args.no_fp16:
        model = model.half()

    index = src.index.Indexer(args.projection_size, args.n_subquantizers, args.n_bits)

    # index all passages
    # for emb_path in args.passages_embeddings:
    input_paths = []
    for emb_path in args.passages_embeddings:
        input_paths += glob.glob(emb_path)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = []
    for passage_path in args.passages:
        print(passage_path)
        passages += src.data.load_passages(passage_path)
    passage_id_map = {x["_id"]: x for x in passages}

    data_paths = glob.glob(args.data)
    alldata = []

    for path in data_paths:
        data = load_data(path)
        qrels = _load_qrels(args.qrels)
        data = [item for item in data if item["_id"] in qrels]
        output_path = os.path.join(args.output_dir, os.path.basename(path))
        queries = [ex["text"] for ex in data]

        if args.prompt is not None:
            prompted_queries = ["{0} [SEP] {1}".format(text, args.prompt) for text in queries]
            questions_embedding = embed_queries(args, prompted_queries, model, tokenizer)
        
        # this is a hack for lotte
        else:
            if args.prompt is None and " </s> " in queries[0]:
                processed_queries = [query.split(" </s> ")[1] for query in queries]
                print(processed_queries[0])
                questions_embedding = embed_queries(args, processed_queries, model, tokenizer)
            if args.prompt is None and " [SEP] " in queries[0]:
                processed_queries = [query.split(" [SEP] ")[1] for query in queries]
                print(processed_queries[0])
                questions_embedding = embed_queries(args, processed_queries, model, tokenizer)

            else:
                questions_embedding = embed_queries(args, queries, model, tokenizer)

        # get top k results
        start_time_retrieval = time.time()
        print(args.n_docs)
        top_ids_and_scores = index.search_knn(questions_embedding, args.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f}")

        # evaluate the scores
        results = add_passages(data, passage_id_map, top_ids_and_scores)
        metrics = {}
        ndcg, _map, recall, precision = evaluate(qrels, results, [1,3,5,10,100,1000])
        for metric  in (ndcg, _map, recall, precision):
            for key, value in metric.items():
                metrics[key] = value
        metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}
        print(metrics)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
        output_path_score = os.path.join(args.output_dir, os.path.basename(path) + "be_score")

        with open(output_path_score, "w") as outfile:
            json.dump(metrics, outfile)

        print(metrics)
        print("saved be evaluation scores")
        
        print("running ce model")
        if args.ce_model is not None:
            ranker = Rerank(args.ce_model)
            reranked_results = ranker.rerank(passage_id_map, data, results, top_k=args.n_docs, prompt=args.ce_prompt)

            add_passages(data, passage_id_map, top_ids_and_scores)

            reranked_metrics = {}
            ndcg, _map, recall, precision = evaluate(qrels, reranked_results, [1,3,5,10,100,1000])
            for metric in (ndcg, _map, recall, precision):
                for key, value in metric.items():
                    reranked_metrics[key] = value
            reranked_metrics = {key: 100 * np.mean(value) for key, value in reranked_metrics.items()}
            print(reranked_metrics)
            ce_output_path_score = os.path.join(args.output_dir, os.path.basename(path) + "ce_score_{}".format(args.ce_prompt.replace(" ", "_") if args.ce_prompt is not None else "null"))
           
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(ce_output_path_score, "w") as outfile:
                json.dump(reranked_metrics, outfile)

            for i, d in tqdm(enumerate(data)):
                reranked_scores = reranked_results[d["_id"]]
                for doc_id, score in reranked_scores.items():
                    for ctx in d["ctxs"]:
                        if ctx["id"] == doc_id:
                            ctx["reranked_score"] = score

            print("saved reranked scores")
        if args.lotte is True:
            final_results = reranked_results if args.ce_model is not None else results
            qid2ranks = {}
            for qid in final_results:
                sorted_results = sorted(final_results[qid].items(), key=lambda x: x[1], reverse=True)
                qid2ranks[int(qid)] = []
                for rank, item in enumerate(sorted_results):
                    qid2ranks[int(qid)].append((qid, item[0], rank + 1, item[1]))
            
            lotte_output_path = os.path.join(args.output_dir, "pooled.search.ranking.tsv")
            print("saving results at")
            print(lotte_output_path)
            with open(lotte_output_path, "w", newline='') as fout:
                tsv_output = csv.writer(fout, delimiter='\t')
                for i in range(1, len(qid2ranks)):
                    result = qid2ranks[i]
                    for p_score in result:
                        tsv_output.writerow([p_score[0], p_score[1], p_score[2], p_score[3]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)", nargs="+")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages", nargs="+")
    parser.add_argument("--qrels", type=str, default=None, help="Glob path to qrel file")
    parser.add_argument("--prompt", type=str, default=None, help="prompt")
    parser.add_argument("--ce_prompt", type=str, default=None, help="prompt for CE")
    parser.add_argument("--ce_model", type=str, default=None, help="renraking model")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--lotte", action="store_true", help="saving lotte format results")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
