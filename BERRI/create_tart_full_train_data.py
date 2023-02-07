# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import argparse
import json
import glob
import random
import jsonlines
import pandas as pd

from tqdm import tqdm


def load_jsonlines(file_name):
    with jsonlines.open(file_name, 'r') as jsonl_f:
        data = [obj for obj in jsonl_f]
    return data


def process_data(input_data, prompts):
    new_data = []
    false_negatives = 0
    for item in input_data:
        query = item["question"]
        positive_ctxs = item["positive_ctxs"]
        if len(positive_ctxs) > args.num_positive_ctxs:
            positive_ctxs = random.sample(
                positive_ctxs, k=args.num_positive_ctxs)
        negative_ctxs = item["negative_ctxs"] + \
            item["hard_negative_ctxs"] if "hard_negative_ctxs" in item else item["negative_ctxs"]
        final_negatives = []
        for neg in negative_ctxs:
            if neg["text"] in [pos["text"] for pos in item["positive_ctxs"]]:
                false_negatives += 1
                continue
            else:
                final_negatives.append(neg)
        negative_ctxs = final_negatives

        if len(negative_ctxs) > args.num_negative_ctxs:
            negative_ctxs = random.sample(
                negative_ctxs, k=args.num_negative_ctxs)

        for pos in positive_ctxs:
            prompt = random.sample(prompts, k=1)[0]
            prompted_query = "{0} </s> {1}".format(prompt, query)

            title = pos["title"]
            text = pos["text"]
            if title is not None and len(title) > 0:
                text = "{0} {1}".format(title, text)
            new_data.append(
                {"query": prompted_query, "document": text, "label": 1})
        for neg in negative_ctxs:
            prompt = random.sample(prompts, k=1)[0]
            prompted_query = "{0} </s> {1}".format(prompt, query)

            title = neg["title"]
            text = neg["text"]
            if title is not None and len(title) > 0:
                text = "{0} {1}".format(title, text)
            new_data.append(
                {"query": prompted_query, "document": text, "label": 0})
    print("{} data created".format(len(new_data)))
    print("false negatives {}".format(false_negatives))
    return new_data


def process_instruction_unfollowing_sample(input_data, prompts, full_data_num):
    inst_num = min(int(full_data_num * 0.8 * 0.2), 10000)
    if len(input_data) > 7500:
        input_data = random.sample(input_data, k=inst_num)
    new_data = []
    for item in input_data:
        query = item["question"]
        sampled_context = random.sample(item["ctxs"], k=1)
        for ctx in sampled_context:
            prompt = random.sample(prompts, k=1)[0]
            prompted_query = "{0} </s> {1}".format(prompt, query)

            title = ctx["title"]
            text = ctx["text"]
            if len(title) > 0:
                text = "{0} {1}".format(title, text)
            new_data.append(
                {"query": prompted_query, "document": text, "label": 0})
    print("instructions unfollowing samples")
    print("{} data created".format(len(new_data)))
    return new_data


def process_prompts(row):
    # load instructions
    prompts = []
    for i in range(1, 9):
        column_name = "prompt_{}".format(i)
        if type(column_name) == str:
            prompts.append(row[column_name])
    return prompts


def load_data(data_path):
    data = []
    with open(data_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            data.append(example)
    return data


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    all_data = []
    file_names = pd.read_csv(args.input_file, sep="\t")
    if args.instruction_unfollowing_file is not None:
        inst_unfollow_dict = {}
        inst_unfollow_file_names = open(
            args.instruction_unfollowing_file).read().split("\n")[:-1]
        for file in inst_unfollow_file_names:
            src_task_name = (file.split("src_")[1]).split("_tgt")[0]
            inst_unfollow_dict.setdefault(src_task_name, [])
            inst_unfollow_dict[src_task_name].append(file)

    if args.sample_dataset_num is not None:
        dataset_names = file_names["dataset"]
        sampled_datasets = random.sample(
            list(dataset_names), k=args.sample_dataset_num)

    for idx, file_data in tqdm(file_names.iterrows()):
        if args.task_names is not None and file_data["dataset"] not in args.task_names:
            continue
        if args.start_idx is not None and idx < args.start_idx:
            continue
        if args.end_idx is not None and idx > args.end_idx:
            continue

        if os.path.exists(os.path.join(args.input_dir, file_data["dataset"] + ".jsonl")) is False:
            print("file name is mssing ")
            print(os.path.join(args.input_dir,
                  file_data["dataset"] + ".jsonl"))
            print(file_data["dataset"])
            continue

        if args.sample_dataset_num is not None and file_data["dataset"] not in sampled_datasets:
            continue

        task_input_file = os.path.join(
            args.input_dir, file_data["dataset"] + ".jsonl")
        print(task_input_file)
        prompts = process_prompts(file_data)
        input_data = load_data(task_input_file)
        processed_training_data = process_data(input_data, prompts)
        all_data += processed_training_data

        if args.instruction_unfollowing_file is not None and file_data["dataset"] in inst_unfollow_dict:
            for unfollowing_file_name in inst_unfollow_dict[file_data["dataset"]]:
                unfollowing_file_name = glob.glob(
                    unfollowing_file_name + "/*.json*")[0]
                print(unfollowing_file_name)
                unfollowing_input_data = load_data(unfollowing_file_name)
                processed_training_data_unfollowing = process_instruction_unfollowing_sample(
                    unfollowing_input_data, prompts, len(processed_training_data))
                all_data += processed_training_data_unfollowing

    if len(all_data) > args.max_train_data:
        all_data = random.sample(all_data, k=args.max_train_data)

    random.shuffle(all_data)
    train_data, dev_data = all_data[10000:], all_data[:10000]

    with jsonlines.open(os.path.join(args.output_dir, "tart_full_train.json"), 'w') as writer:
        writer.write_all(train_data)
    with jsonlines.open(os.path.join(args.output_dir, "tart_full_dev.json"), 'w') as writer:
        writer.write_all(dev_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--sample_dataset_num", type=int, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--instruction_unfollowing_file", type=str, default=None,
                        help="instruction unfollowing file")
    parser.add_argument("--prompts", type=str, default=None,
                        help="prompt", nargs="+")
    parser.add_argument("--task_names", type=str, default=None,
                        help="task names filter", nargs="+")
    parser.add_argument("--instance_idx_start", type=int,
                        default=None, help="instance start index")
    parser.add_argument("--instance_idx_end", type=int,
                        default=None, help="instance end index")
    parser.add_argument("--per_gpu_batch_size", type=int,
                        default=64, help="Batch size for question encoding")
    parser.add_argument("--n_docs", type=int, default=100,
                        help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true",
                        help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int,
                        default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--start_idx", type=int, default=None,)
    parser.add_argument("--end_idx", type=int, default=None,)
    parser.add_argument("--max_train_data", type=int, default=3000000)
    parser.add_argument("--num_positive_paragraphs", type=int, default=1,)
    parser.add_argument("--num_negative_paragraphs", type=int, default=4,)
    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)
