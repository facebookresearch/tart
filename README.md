# Task-aware Retrieval with Instructions

This is the official repository for our preprint, [Task-aware Retrieval with Instructions](https://arxiv.org/abs/2211.09260). 

We introduce a new retrieval task formulation, **retrieval with instructions**, constructs **BERRI**, the first large-scale collection of retrieval datasets with instructions, and present **TART**, multi-task instruction-following retrieval models trained on BERRI. 

![tart](figures/intro.png)

## Content

1. [Getting started](#getting-started)
2. [Pretrained Checkpoints](#pre-trained-checkpoints)
3. [Evaluation](#evaluation)
    - [BERRI](#beir)
    - [LOTTE](#lotte)
    - [Cross-task Cross-domain evaluation](#cross-task-cross-domain-dataset)
4. [Training](#training)
5. [Dataset: BERRI](#dataset-berri)
5. [Citations and Contact](#citation-and-contact)


## Getting started
First, please download the pretrained models. See the list of the [pretrained checkpoints](). You can also download [preencoded embeddings]() and run interactive evaluations, as shown in the [Interactive mode]() setion. 

```sh
cd TART
wget https://homes.cs.washington.edu/~akari/tart/tart_full_flan_t5_xl.zip
unzip tart_full_flan_t5_xl.zip
```
Then you can test TART as follows: 

```py
from src.enc_t5 import EncT5ForSequenceClassification, EncT5Tokenizer
import torch
import torch.nn.functional as F

# load TART full and tokenizer
model = EncT5ForSequenceClassification.from_pretrained("tart_full_flan_t5_xl")
tokenizer =  EncT5Tokenizer.from_pretrained("tart_full_flan_t5_xl")
model.eval()

# instructions and examples 
in_a = "retrieve a passage that answers this question from Wikipedia"
in_b = "retrieve a question that is similar to the following"
p = "The population of Japan's capital dropped by about 48,600 people to just under 14 million at the start of 2022, the first decline since 1996, the metropolitan government reported Monday."
q_1 = 'How many people live in Tokyo?'
q_2 = 'How many people live in Berlin?'
features = tokenizer(['{0} [SEP] {1}'.format(in_b, q), '{0} [SEP] {1}'.format(in_b, q), '{0} [SEP] {1}'.format(in_b, q), '{0} [SEP] {1}'.format(in_a, q)], [q_1, q_2, p, p], padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    scores = model(**features).logits
    normalized_scores = F.softmax(scores, dim=1)
    print(normalized_scores) 

>> tensor([[0.0012, 0.9988],
        [0.9202, 0.0798],
        [0.3136, 0.6864],
        [0.0770, 0.9230]]) 
```
As you can see, TART not only gives lower scores to the wrong query (q_1 v.s. q_2), but also gives a lower score to the document that is relevant but does not follow the instruction. 


### Interactive mode
Interactive mode enables you to type whatever questions and retrieve pre-encoded documents. To run this interactive mode, you need to encode documents, as well as the models. 

```sh
python interactive.py \
--passages scifact/corpus.jsonl \
--passages_embeddings "scifact_contriever_output_embeddings/*" \
--model_name_or_path facebook/contriever-msmarco \
--ce_model t5-base-lm-adapt_full_new_denoised_data/checkpoint-6000 \
```


## Pre-trained checkpoints 
### TART-full
We release TART-full models trained on BERRI using different initial encoder weights. Our TART-full model on the paper is based on T0-3B. 

| name      | size | initialization | 
| ----------- | ----------- | ----------- |
| TART-full-T0-3b (TART-full in the paper)     |   1.5 billions     |[T0-3B](https://huggingface.co/bigscience/T0_3B)|
| TART-full-T5-lm-adapt-xl   |     1.5 billions    | [T5-xl-LM-adapt](https://huggingface.co/google/t5-xl-lm-adapt)|
| TART-full-T5-lm-adapt-large   |    340 millions     | [T5-large-LM-adapt](https://huggingface.co/google/t5-large-lm-adapt) |
|  TART-full-FLANT5-3B  |  1.5 billions | [FLANT5-XL](https://huggingface.co/google/flan-t5-xl)|
|  TART-full-FLANT5-large  |  340 millions         | [FLANT5-large](https://huggingface.co/google/flan-t5-large)| 
|  TART-full-TkInstruct-3B  | 1.5 billions        | [TkInstruct-3B](allenai/tk-instruct-3b-def-pos)|
|  TART-full-TkInstruct-large  | 340 millions            | [TkInstruct-large](allenai/tk-instruct-large-def-pos)|

### TART-dual 
TART-dual is an efficient bi-encoder model sharing an encoder for document and query encodings. The main model on the paper uses [Contriever-MS MARCO](facebook/contriever-msmarco) pre-trained on Wikipedia 2020 dump. 
We also release TART-dual trained on different base models of [GTR](https://arxiv.org/abs/2112.07899). 


## Evaluation
### BEIR
You can evaluate the models on BEIR, by running `eval_beir.py` or `eval_cross_task.py`. 

`eval_beir.py` is adopted from the official BEIR repository, encodes and runs inference using a single GPU every time, while `eval_cross_task.py` assumes that you have encoded document embeddings and parallelize inference using multiple GPUs. If you have multiple GPUs or try to evaluate TART on datasets with millions of documents (e.g., Climate-FEVER), we recommend using `eval_cross_task.py` script. 

#### Run evaluation with `eval_beir.py`

```sh
python eval_beir.py \
    --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH \
    --dataset BEIR_DATASET_NAME \
    --output_dir YOUR_OUTPUT_DIR
    --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH \
    --ce_model CROSS_ENCODER_MODEL_NAME_OR_PATH \
    --prompt  "YOUR INSTRUCTIONS"
```


#### Run evaluation with `eval_cross_task.py`
As mentioned above, there are two steps to run `eval_cross_task.py` script: **STEP1: encode all documents**, and **STEP2: run evaluations using encoded embeddings**. 

##### STEP1: Encode all of the document
To encode document using a single GPU, please run the command below: 

```sh
python generate_passage_embeddings.py --model_name_or_path YOUR_MODEL_NAME --output_dir OUTPUT_DIR_NAME \
    --passages PATH_TO_YOUR_INPUT_DATA_DIR/corpus.jsonl --shard_id ${i} --num_shards 1
```

If you want to use multiple GPUs to speed up the process, you can run the following command: 

```sh
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_passage_embeddings.py --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH --output_dir OUTPUT_DIR_NAME \
      --passages PATH_TO_YOUR_INPUT_DATA_DIR/corpus.jsonl --shard_id ${i}  --num_shards 8 > ./log/nohup.log.${i} 2>&1 &
done
```

The corpus file is a `jsonlines` file, where each item contains `text` and `title`, and optional `_id` and `meta_data`. 

e.g., 

```
{"_id": "doc9", "title": "Chicago Fire (season 4)", "text": "Hermann is rushed to Chicago Med after being stabbed at Molly's. After losing a lot a blood, it is determined he needs emergency surgery. Feeling guilty about Hermann's present state, Cruz searches for Freddy to turn him in. Severide is reinstated as Lieutenant while Borelli grows more concerned about Chili's erratic behavior. Mouch considers finally proposing to Platt.", "metadata": {}}
``` 

##### STEP2: Run predictions 

Once you encode passages, you can run the evaluations as follows: 
```sh
python eval_cross_task.py \
    --passages PATH_TO_YOUR_INPUT_DATA_DIR/corpus.jsonl \
    --passages_embeddings "PATH_TO_YOUR_EMBEDDING_OUTPUT_DIR/passages_*" \
    --qrels PATH_TO_YOUR_INPUT_DATA_DIR/qrels/test.csv  \
    --output_dir OUT_PUT_DIR_NAME \
    --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH \
    --ce_model CROSS_ENCODER_MODEL_NAME_OR_PATH \
    --data PATH_TO_YOUR_INPUT_DATA_DIR/queries.jsonl \
    --prompt  "YOUR INSTRUCTIONS"
```

### LOTTE
We evaluate our model on LOTTE-search (pooled). To run the evaluations on LOTTE, you can download our processed data (the data itself is the same but we convert the input data file formats and add instructions) as follows:

```sh
wget https://homes.cs.washington.edu/~akari/tart/processed_lotte_search_pooled.zip
unzip processed_lotte_search_pooled.zip
```

Encode passages as in the previous section. 

```sh
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_passage_embeddings.py --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH --output_dir OUTPUT_DIR_NAME \
      --passages processed_lotte_search_pooled/corpus.jsonl --shard_id ${i}  --num_shards 8 > ./log/nohup.log.${i} 2>&1 &
done
```

Once you encode the passages, you can run evaluations 
```sh
python eval_cross_task.py \
    --passages processed_lotte_search_pooled/corpus.jsonl \
    --passages_embeddings "contriever_lotte_corpus/passages_*" \
    --qrels processed_lotte_search_pooled/qrels/test.tsv \
    --output_dir OUT_PUT_DIR_NAME \
    --model_name_or_path BI_ENCODER_MODEL_NAME_OR_PATH \
    --ce_model CROSS_ENCODER_MODEL_NAME_OR_PATH \
    --data processed_lotte_search_pooled/queries_w_instructions_sep.jsonl \
    --ce_model CROSS_ENCODER_MODEL_NAME_OR_PATH --lotte
```
This code output the lotte's official evaluation script format data under `CROSS_ENCODER_MODEL_NAME_OR_PATH/`
Then you can run the official evaluation script as follows: 

```sh
cp lotte
python evaluate_lotte_rankings.py --k 5 --split test --data_path  ../lotte --rankings_path PATH_TO_PREDICTION_FILE
```

### Cross-task Cross-domain dataset 
In this paper, we newly introduce cross-task cross-domain evaluation, where given an instruction and a single large-scale domain, a system needs to retrieve documents that follow instructions. 

Due to legal reasons, Meta cannot host this data. The script to create cross-task cross-domain dataset is available at [cross_task_cross_domain](), and you can also download the processed cross task dataset as follows.

```sh
wget https://homes.cs.washington.edu/~akari/tart/cross_task_cross_domain_final.zip
unzip https://homes.cs.washington.edu/~akari/tart/cross_task_cross_domain_final.zip
```

Due to the larger corpus, we highly recommend encoding every documents beforehand. 
Encoded documents are available at the [encoded documents]() Section. 

Then you can run evaluations on the cross-task cross-domain data as follows: 
```sh
python eval_cross_task.py \
    --passages nq/corpus.jsonl scifact/corpus.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/gooaq_med/corpus_new.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/linkso_py/corpus_new.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/ambig/corpus_new.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/wikiqa/corpus_new.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/gooaq_technical/corpus_new.jsonl /private/home/akariasai/inst_dpr/cross_task_eval/codesearch_py/corpus_new.jsonl \
--passages_embeddings "fixed_linkso_py_contriever/passages_*" "ambig_fixed_contriever_embeddings/passages_*" "scifact_contriever_output_embeddings/*" "contriever_msmarco_embeddings_nq/passages_*" "gooaq_technical_fixed_contriever_embeddings/passages_*" "fixed_codesearch_py_contriever/passages_*" "wikiqa_fixed_contriever_embeddings/passages_*" "fixed_gooaq_medical/passages_*" \
--qrels /private/home/akariasai/inst_dpr/cross_task_eval/ambig/qrels/test_new.tsv \
--output_dir ambig_cross_task_fixed_contriever_ce_rerabk \
--model_name_or_path facebook/contriever-msmarco \
--data /private/home/akariasai/inst_dpr/cross_task_eval/ambig/queries.jsonl \
--ce_model cross-encoder/ms-marco-MiniLM-L-12-v2
```

## Training 
Coming soon! 


## Dataset: BERRI

Due to legal reasons, Meta cannot host reproduced Wikidata. We include scripts to reproduce BERRI by downloading checkpoints, converting the inputs and running models to collect positive and negative datasets. See the details at [BERRI](BERRI). 

Alternatively, you can download data from a third party, which has been produced using scripts from this directory. 


## Citation and Contact 
If you find this repository helpful, please cite our paper. 

```
@article{asai2022tart,
  title={Task-aware Retrieval with Instructions},
  author={Asai, Akari and Schick, Timo and Lewis, Patrick and Chen, Xilun and Izacard, Gautier and Riedel, Sebastian and Hajishirzi, Hannaneh and Yih, Wen-tau},
  journal={arXiv preprint arXiv:2211.09260},
  year={2022}
}
```

If you have any questions about the paper, feel free to contact Akari Asai (akari[at]cs.washington.edu) or open an issue, and mention @AkariAsai