# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.modeling_enc_t5 import EncT5ForSequenceClassification
from src.tokenization_enc_t5 import EncT5Tokenizer

logger = logging.getLogger(__name__)

#Parent class for any reranking model
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
        self.model.to('cuda')

        self.model.eval()

    def rerank(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str],
               results: Dict[str, Dict[str, float]],
               top_k: int, 
               prompt: str = None) -> Dict[str, Dict[str, float]]:
        
        sentence_pairs, pair_ids = [], []
        
        self.rerank_results = {query_id: {} for query_id in results}
        for query_id in tqdm(results):
            docs = []
            query= []
            doc_ids = []

            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    # print(corpus_text)
                    # print(corpus_text)
                    docs.append(corpus_text)
                    doc_ids.append(doc_id)
                    if prompt is None:
                        # sentence_pairs.append([queries[query_id], corpus_text])
                        query.append(queries[query_id])
                    else:
                        # sentence_pairs.append(["{0} [SEP] {1}".format(prompt, queries[query_id]), corpus_text])
                        query.append("{0} [SEP] {1}".format(prompt, queries[query_id]))
                # print(query[-1])
            
            else:
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (corpus[doc_id].get("title", "") + " " + corpus[doc_id].get("text", "")).strip()
                    # print(corpus_text)
                    docs.append(corpus_text)
                    doc_ids.append(doc_id)
                    if prompt is None:
                        query.append(queries[query_id])
                        # sentence_pairs.append([queries[query_id], corpus_text])
                    else:
                        query.append("{0} [SEP] {1}".format(prompt, queries[query_id]))
                        # sentence_pairs.append(["{0} [SEP] {1}".format(prompt, queries[query_id]), corpus_text])

            # run inference 
            features = self.tokenizer(query, docs, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
            with torch.no_grad():
                scores = self.model(**features).logits
                normalized_scores = F.softmax(scores, dim=1)
            final_scores = [float(score[1]) for score in normalized_scores]
            # print(final_scores)
            for doc_id, score in zip(doc_ids, final_scores):
                self.rerank_results[query_id][doc_id] = score

        # #### Starting to Rerank using cross-attention
        # logging.info("Starting To Rerank Top-{}....".format(top_k))
        
        # rerank_scores = [float(score[1]) for score in self.cross_encoder.predict(sentence_pairs, batch_size=self.batch_size)]
        # #### Reranking results
        # self.rerank_results = {query_id: {} for query_id in results}
        # for pair, score in zip(pair_ids, rerank_scores):
        #     query_id, doc_id = pair[0], pair[1]
        #     self.rerank_results[query_id][doc_id] = score
        # print(self.rerank_results)
        return self.rerank_results