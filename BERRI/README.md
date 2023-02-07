# BERRI

**BERRI** is a collection of retrieval task with instructions. 

Due to some legal reasons, Meta cannot directly release the preprocessed scripts, so this repository contains the script to re-process and create data.   
You can also download the data processed by third party below:

You can download the processed source data (from the process (i)) as well as the final training data for TART-dual and full, processed by a third party here:
- [source data (22 GB)](https://drive.google.com/file/d/1hzlN4cEFOZRkdVeCMq62NUxvMNTopB1o/view?usp=share_link) 
- [TART-full training data (1 GB)](https://drive.google.com/file/d/1oijzAb2gWKT54OgeE7_KB9VcHvA7UxpQ/view?usp=share_link)
- [TART-dual training data (14 GB)](https://drive.google.com/file/d/1lMmD5lTxYWYf0z0ua0-GaGKz2qs2mG1r/view?usp=share_link)


## Preprocessing
First please download the corpus (`corpus.tsv`) and source data file (`qa.jsonl`) for all retrieval tasks [here]((https://drive.google.com/file/d/1hzlN4cEFOZRkdVeCMq62NUxvMNTopB1o/view?usp=share_link). 


### Step 1: run Contriever to find the top documents

- generate embeddings
First, generate passage embeddings using `facebook/contriever-msmarco`.

```sh
cd ../TART
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_passage_embeddings.py --model_name_or_path facebook/contriever-msmarco --output_dir OUTPUT_DIR_NAME \
      --passages ../BERRI/berri_corpus_data/TASK_NAME/corpus.tsv --shard_id ${i}  --num_shards 8 > ./log/nohup.log.${i} 2>&1 &
done
```

Then, retrieve top passages as follows:

```
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages ../BERRI/berri_corpus_data/TASK_NAME/corpus.tsv \
    --passages_embeddings "YOUR_EMBEDDING_PATH/*" \
    --data ../BERRI/berri_corpus_data/TASK_NAME/qa_data.json \
    --output_dir PATH_TO_OUTPUT_DIR --n_docs 100
```

### Step 2: Denoise the passages

```
python denoising.py \
    --task_name TASK_NAME \
    --train_file PATH_TO_TRAIN_FILE \
    --test_input_file output_dir/qa_data.json \
    --model_name_or_path PATH_TO_DENOISING_MODEL_NAME \
    --output_dir PATH_TO_OUTPUT_DIR \
    --do_predict \
    --evaluation_strategy steps \
    --max_seq_length 512 --overwrite_cache --top_k 30 \
    --instruction_file berri_instructions.tsv # only for creating tart-dual training data. 
```

### Step 3: Combine denoised results and create training data

- Creating TART-dual training data
```
python create_tart_dual_train_data.py \
    --inst_file berri_instructions.tsv \
    --output_file PATH_TO_OUTPUT_DIR \
    --input_dir PATH_TO_DENOISED_RESULTS \
```
- Creating TART-full training data
```
python create_tart_full_train_data.py \
    --input_file berri_instructions.tsv \
    --output_dir --output_file PATH_TO_OUTPUT_DIR \
    --input_dir PATH_TO_DENOISED_RESULTS \
```



