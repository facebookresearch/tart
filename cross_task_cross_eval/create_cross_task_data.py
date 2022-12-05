import pandas as pd
import collections
import numpy
import json
import jsonlines
import csv
import os
import random
import tqdm
import datasets
wikiqa = datasets.load_dataset("wiki_qa")

def load_jsonlines(file_name):
    with jsonlines.open(file_name, 'r') as jsonl_f:
        data = [obj for obj in jsonl_f]
    return data


os.mkdir("ambig")
os.mkdir("wikiqa")
os.mkdir("gooqa_tech")
os.mkdir("linkso_py")
os.mkdir("codesearch_py")


# WIKIQA
corpus = {}
for split in ["train", "test", "validation"]:
    for item in wikiqa[split]:
        corpus.setdefault(item["document_title"], [])
        if item["answer"] not in corpus[item["document_title"]]:
            corpus[item["document_title"]].append(item["answer"])
    
final_corpus = []
for title in corpus:
    for idx, doc in enumerate(corpus[title]):
        final_corpus.append({"title": title, "text": doc, "_id":"{0}_{1}".format(title, idx), "metadata": {}})

final_qrel_data = []
final_queries = []
for item in wikiqa["validation"]:
    question_id = item["question_id"]
    question = item["question"]
    if item["label"] == 1:
        corpus_id = corpus[item["document_title"]].index(item["answer"])
        final_queries.append({"_id": "wikiqa_{}".format(question_id), "text": question, "metadata": {}})
        final_qrel_data.append({"query-id": "wikiqa_{}".format(question_id), "corpus-id": "{0}_{1}".format(item["document_title"], corpus_id), "score": 1})
for item in wikiqa["test"]:
    question_id = item["question_id"]
    question = item["question"]
    if item["label"] == 1:
        corpus_id = corpus[item["document_title"]].index(item["answer"])
        final_queries.append({"_id": "wikiqa_{}".format(question_id), "text": question, "metadata": {}})
        final_qrel_data.append({"query-id": "wikiqa_{}".format(question_id), "corpus-id": "{0}_{1}".format(item["document_title"], corpus_id), "score": 1})

q2dic = {}
for item in final_queries:
    q2dic[item["_id"]] = item
final_queries = list(q2dic.values())

with jsonlines.open('wikiqa/queries.jsonl', 'w') as writer:
    writer.write_all(final_queries)
with jsonlines.open('wikiqa/corpus.jsonl', 'w') as writer:
    writer.write_all(final_corpus)
with open('wikiqa/qrels/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', "score"])
    for item in final_qrel_data:
        tsv_writer.writerow([item["query-id"], item["corpus-id"], item["score"]])


# Ambig QA
ambigqa_path = "data/ambignq_light/"
ambigqa_dev_data = load_jsonlines(ambigqa_path + "dev_light.json")[0]
ambigqa_train_data = load_jsonlines(ambigqa_path + "train_light.json")[0]

ambig_evals_qq = {}
final_queries = []
final_corpus = []
count = 0
for item in ambigqa_train_data:
    for an in item["annotations"]:
        if an["type"] == "multipleQAs":
            qa_pairs = an["qaPairs"]
            for q in qa_pairs:
                final_corpus.append({"_id": "ambig_train_{0}".format(count), "text": q["question"], "title": "", "metadata": {}})
                count += 1

final_qrels = []
count = 0
for item in ambigqa_dev_data:
    for an in item["annotations"]:
        if an["type"] == "multipleQAs":
            qa_pairs = an["qaPairs"]
            for qa in qa_pairs:
                target_id = "ambig_test_{0}".format(count)
                final_corpus.append({"_id": "ambig_test_{0}".format(count), "text": qa["question"], "title": "" , "metadata": {}})
                final_queries.append({"_id": "ambig_nq_source_{}".format(item["id"]), "text": item["question"], "metadata": {}})
                final_qrels.append({"corpus-id": "ambig_test_{0}".format(count), "query-id": "ambig_nq_source_{}".format(item["id"]), "score": 1})
                count += 1
            
q2dic = {}
for item in final_queries:
    q2dic[item["_id"]] = item
final_queries = list(q2dic.values())

with jsonlines.open('ambig/queries.jsonl', 'w') as writer:
    writer.write_all(final_queries)
with jsonlines.open('ambig/corpus.jsonl', 'w') as writer:
    writer.write_all(final_corpus)
with open('ambig/qrels/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', "score"])
    for item in final_qrels:
        tsv_writer.writerow([item["query-id"], item["corpus-id"], item["score"]])


# GooAQ technical
gooqa_technical = []
gooaq_data = load_jsonlines("data/gooaq.jsonl")

for item in gooaq_data:
    if item["answer_type"] == "feat_snip" and item["answer_url"] is not None and "https://" in item["answer_url"] :
        url = item["answer_url"].split("https://")[1].split("/")[0]
        if url == "stackoverflow.com":
            item["url_processed"] = url
            gooqa_technical.append(item)

random_sampled_qooaq_tech = random.sample(gooqa_technical, k=1000)

full_corpus = []
full_queries = []
full_qrels = []
answer2id = {}

for idx, item in enumerate(gooqa_technical):
    full_corpus.append({"_id": "{0}_{1}".format(item["url_processed"], idx), "text": item["answer"], "title": "", "metadata": {}})
    answer2id[item["answer"]] = "{0}_{1}".format(item["url_processed"], idx)


for item in random_sampled_qooaq_tech:
    full_queries.append({"_id": "gooaq_technical_{}".format(item["id"]), "text": item["question"], "metadata": {}})
    corpus_id = answer2id[item["answer"]]
    full_qrels.append({"query-id": "gooaq_technical_{}".format(item["id"]), "corpus-id": corpus_id, "score": 1})


os.mkdir("gooaq_technical/qrels")
with jsonlines.open('gooaq_technical/queries.jsonl', 'w') as writer:
    writer.write_all(full_queries)
with jsonlines.open('gooaq_technical/corpus.jsonl', 'w') as writer:
    writer.write_all(full_corpus)
with open('gooaq_technical/qrels/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', "score"])
    for item in full_qrels:
        tsv_writer.writerow([item["query-id"], item["corpus-id"], item["score"]])
        

# LinkSO
def find_duplicated_qestions(dir, lang):
    duplicated_q_pairs = []
    qid2all = pd.read_csv(os.path.join(dir, "{}_qid2all.txt".format(lang)), sep="\t", header=None)
    qid2all_dic = {}
    for idx, row in qid2all.iterrows():
        qid2all_dic[int(row[0])] = {"title": row[1], "body": row[2]}

    cosin =  pd.read_csv(os.path.join(dir, "{}_cosidf.txt".format(lang)), sep="\t")
    dup_pair_ids = {}
    for idx, row in cosin.iterrows():
        if row["label"] == 1:
            dup_pair_ids[int(row["qid1"])] = int(row["qid2"])
        
    test_qs = open(os.path.join(dir, "{}_test_qid.txt".format(lang))).read().split("\n")[:-1]
    for q_id in test_qs:
        if int(q_id) in dup_pair_ids:
            dup_id = dup_pair_ids[int(q_id)]
            duplicated_q_pairs.append((qid2all_dic[int(q_id)], qid2all_dic[dup_id]))
    return duplicated_q_pairs

linkso_data_python = "/private/home/akariasai/inst_dpr/preprocessing/linkso_data/topublish/python"
full_corpus, full_queries, full_qrels = find_duplicated_qestions(linkso_data_python, "python")
linkso_dups_python = find_duplicated_qestions(linkso_data_python, "python")
qid2queries = {item["_id"]: item["text"] for item in full_queries}
qid2corpus = {item["_id"]: item for item in full_corpus}

with jsonlines.open('linkso_py/queries.jsonl', 'w') as writer:
    writer.write_all(full_queries)
with jsonlines.open('linkso_py/corpus.jsonl', 'w') as writer:
    writer.write_all(full_corpus)
with open('linkso_py/qrels/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', "score"])
    for item in full_qrels:
        tsv_writer.writerow([item["query-id"], item["corpus-id"], item["score"]])

# CodeSearch Net Py
python_code_serach_net = datasets.load_dataset("code_search_net", "python")
python_short_descs = [item for item in python_code_serach_net["test"] if len(item["func_documentation_string"]) < 300 and len(item["func_documentation_string"]) > 50]

full_corpus = []
full_queries = []
full_qrels = []
answer2id = {}

for idx, item in tqdm(enumerate(python_code_serach_net["train"])):
    doc_id = "codeserachnet_python_train_{0}_{1}".format(idx, item["func_name"])
    if '"""' in item["func_code_string"]:
        code = (item["func_code_string"].split('"""')[0] +  item["func_code_string"].split('"""')[2]).replace("\n\n", "")
    elif "'''" in item["func_code_string"]:
        code = (item["func_code_string"].split("'''")[0] +  item["func_code_string"].split("'''")[2]).replace("\n\n", "")
    else:
        code = item["func_code_string"]
    full_corpus.append({"_id": doc_id, "text": code, "metadata": {}, "title": "" })
    answer2id[code]  =  doc_id

for idx, item in tqdm(enumerate(python_code_serach_net["validation"])):
    doc_id = "codeserachnet_python_validation_{0}_{1}".format(idx, item["func_name"])
    if '"""' in item["func_code_string"]:
        code = (item["func_code_string"].split('"""')[0] +  item["func_code_string"].split('"""')[2]).replace("\n\n", "")
    elif "'''" in item["func_code_string"]:
        code = (item["func_code_string"].split("'''")[0] +  item["func_code_string"].split("'''")[2]).replace("\n\n", "")
    else:
        code = item["func_code_string"]
    full_corpus.append({"_id": doc_id, "text": code, "metadata": {}, "title": "" })
    answer2id[code]  =  doc_id

for idx, item in tqdm(enumerate(python_code_serach_net["test"])):
    doc_id = "codeserachnet_python_test_{0}_{1}".format(idx, item["func_name"])
    if '"""' in item["func_code_string"]:
        code = (item["func_code_string"].split('"""')[0] +  item["func_code_string"].split('"""')[2]).replace("\n\n", "")
    elif "'''" in item["func_code_string"]:
        code = (item["func_code_string"].split("'''")[0] +  item["func_code_string"].split("'''")[2]).replace("\n\n", "")
    else:
        code = item["func_code_string"]
    full_corpus.append({"_id": doc_id, "text": code, "metadata": {}, "title": "" })
    answer2id[code]  =  doc_id

random_sampled_python_short_descs= random.sample(python_short_descs, k = 1000)

for idx, item in enumerate(random_sampled_python_short_descs):
    qid = "python_codesearch_{}".format(idx)
    
    query = item["func_documentation_string"]
    if '"""' in item["func_code_string"]:
        code = (item["func_code_string"].split('"""')[0] +  item["func_code_string"].split('"""')[2]).replace("\n\n", "")
    elif "'''" in item["func_code_string"]:
        code = (item["func_code_string"].split("'''")[0] +  item["func_code_string"].split("'''")[2]).replace("\n\n", "")
    else:
        code = item["func_code_string"]
    corpus_id = answer2id[code]
    full_queries.append({"_id": qid, "text": query, "metadata": {}})
    full_qrels.append({"corpus-id": corpus_id, "query-id": qid, "score": 1})
    
os.mkdir("codesearch_py/qrels")
with jsonlines.open('codesearch_py/queries.jsonl', 'w') as writer:
    writer.write_all(full_queries)
with jsonlines.open('codesearch_py/corpus.jsonl', 'w') as writer:
    writer.write_all(full_corpus)
with open('codesearch_py/qrels/test.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', "score"])
    for item in full_qrels:
        tsv_writer.writerow([item["query-id"], item["corpus-id"], item["score"]])