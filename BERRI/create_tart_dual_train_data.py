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
import numpy as np
import pandas as pd

import numpy as np
from tqdm import tqdm
import copy


def load_data(data_path):
    data = []
    with open(data_path, "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            data.append(example)
    return data


def process_instruction_unfollowing_sample(input_data, prompts):
    new_data = {}
    for item in input_data:
        query = item["question"]
        sampled_context = random.sample(item["ctxs"], k=1)
        new_data[query] = sampled_context
    return new_data


def load_jsonlines(file_name):
    with jsonlines.open(file_name, 'r') as jsonl_f:
        data = [obj for obj in jsonl_f]
    return data


def process_prompts(row):
    # load instructions
    prompts = []
    for i in range(1, 9):
        column_name = "prompt_{}".format(i)
        if type(column_name) == str:
            prompts.append(row[column_name])
    return prompts


def main(args):
    print(f"Loading model from: {args.model_name_or_path}")
    final_data = []
    min_neg_nums = 100
    file_names = pd.read_csv(args.inst_file, sep="\t")

    if args.instruction_unfollowing_file is not None:
        inst_unfollow_dict = {}
        inst_unfollow_file_names = open(
            args.instruction_unfollowing_file).read().split("\n")[:-1]
        # Load instruction unfollowing files.
        # src indicates the original task name (query task) while tgt indicate the corpus task name.
        for file in inst_unfollow_file_names:
            src_task_name = (file.split("src_")[1]).split("_tgt")[0]
            inst_unfollow_dict.setdefault(src_task_name, [])
            inst_unfollow_dict[src_task_name].append(file)

    for _, file_data in tqdm(file_names.iterrows()):
        input_file = os.path.join(
            args.input_dir, file_data["dataset"] + ".jsonl")
        prompts = process_prompts(file_data)
        input_data = load_jsonlines(input_file)
        # mostly for ablation. Skip tasks that are specified in `ignore_tasks`, or that are not specified in `task_names`.
        if args.ignore_tasks is not None and file_data["dataset"] in args.ignore_tasks:
            print("skip {0}".format(file_data["dataset"]))
            continue
        if args.task_names is not None and file_data["dataset"] not in args.task_names:
            print("skip {0}".format(file_data["dataset"]))
            continue
        print("# of data: {}".format(len(input_data)))

        # add instruction-unfollowing data.
        if args.instruction_unfollowing_file is not None and file_data["dataset"] in inst_unfollow_dict:
            for unfollowing_file_name in inst_unfollow_dict[file_data["dataset"]]:
                unfollowing_file_name = glob.glob(
                    unfollowing_file_name + "/*.json*")[0]
                print(unfollowing_file_name)
                unfollowing_input_data = load_data(unfollowing_file_name)
                processed_training_data_unfollowing = process_instruction_unfollowing_sample(
                    unfollowing_input_data, prompts)

        for datapoint in input_data:
            if args.instruction_unfollowing_file is not None and file_data["dataset"] in inst_unfollow_dict and datapoint["question"] in processed_training_data_unfollowing:
                instructions_unfollowing_negatives = processed_training_data_unfollowing[
                    datapoint["question"]]
            else:
                instructions_unfollowing_negatives = []
            if len(datapoint["question"]) < 20:
                continue

            # To make a model robust to different instructions, sample two prompts per instance.
            if len(prompts) > 2:
                sampled_prompts = random.sample(prompts, k=2)
            else:
                sampled_prompts = prompts
            true_negatives = []
            true_hard_negatives = []
            # final check if the paragraph is not included in the labeled gold paragraphs.
            for neg in datapoint["negative_ctxs"]:
                skip = False
                for pos in datapoint["positive_ctxs"]:
                    if neg["text"] == pos["text"]:
                        print("false negatives")
                        skip = True
                if skip is False:
                    true_negatives.append(neg)

            # a cross encoder can falsely predict a positive paragraph "negative".
            # check if the paragraph is not included in the labeled gold paragraphs.
            for neg in datapoint["hard_negative_ctxs"]:
                skip = False
                for pos in datapoint["positive_ctxs"]:
                    if neg["text"] == pos["text"]:
                        print("false negatives")
                        skip = True
                if skip is False:
                    true_hard_negatives.append(neg)
            datapoint["negative_ctxs"] = true_negatives
            datapoint["hard_negative_ctxs"] = true_hard_negatives

            # create training samples.

            for p in sampled_prompts:
                new_data = copy.deepcopy(datapoint)
                new_data["question"] = "{0} [SEP] {1}".format(
                    p, new_data["question"])
                if args.only_negative is True:
                    # do not add additional CE scores
                    new_data["positive_ctxs"] = [
                        pos for pos in new_data["positive_ctxs"] if "ce_score" not in pos]

                if len(new_data["positive_ctxs"]) > args.num_positive_paragraphs:
                    new_data["positive_ctxs"] = random.sample(
                        new_data["positive_ctxs"], k=args.num_positive_paragraphs)
                    for pos in new_data["positive_ctxs"]:

                        if "title" not in pos:
                            neg["title"] = ""

                        if pos["title"] is None:
                            print("none title")
                            pos["title"] = ""

                        if type(pos["text"]) is list:
                            pos["text"] = pos["text"][0]

                if len(new_data["negative_ctxs"]) > args.num_negative_paragraphs:
                    new_data["negative_ctxs"] = random.sample(
                        new_data["negative_ctxs"], k=args.num_negative_paragraphs)
                    for neg in new_data["negative_ctxs"]:
                        if type(neg["text"]) is list:
                            neg["text"] = neg["text"][0]

                        if "title" not in neg:
                            neg["title"] = ""

                        if neg["title"] is None:
                            neg["title"] = ""

                if len(new_data["hard_negative_ctxs"]) > args.num_hard_negative_paragraphs:
                    new_data["hard_negative_ctxs"] = random.sample(
                        new_data["hard_negative_ctxs"], k=args.num_hard_negative_paragraphs)
                    for neg in new_data["hard_negative_ctxs"]:
                        if type(neg["text"]) is list:
                            neg["text"] = neg["text"][0]
                        if "title" not in neg:
                            neg["title"] = ""
                        if neg["title"] is None:
                            neg["title"] = ""

                if len(instructions_unfollowing_negatives) > args.num_instructions_unfollowing_negatives:
                    new_data["hard_negative_ctxs"] = random.sample(
                        new_data["hard_negative_ctxs"], k=args.num_instructions_unfollowing_negatives)
                    for neg in instructions_unfollowing_negatives:
                        if type(neg["text"]) is list:
                            neg["text"] = neg["text"][0]
                        if "title" not in neg:
                            neg["title"] = ""
                        if neg["title"] is None:
                            neg["title"] = ""
                        new_data["hard_negative_ctxs"].append(neg)

                assert len(new_data["positive_ctxs"]) > 0
                final_data.append(new_data)
                if len(new_data["negative_ctxs"]) + len(new_data["hard_negative_ctxs"]) < min_neg_nums:
                    min_neg_nums = len(
                        new_data["negative_ctxs"]) + len(new_data["hard_negative_ctxs"])
        print("# of data: {}".format(len(final_data)))

    random.shuffle(final_data)
    # split data into train and dev set for development purpose.
    train_data, dev_data = final_data[5000:], final_data[:5000]

    with jsonlines.open(args.output_file + "_train.jsonl", "w") as writer:
        writer.write_all(train_data)
    with jsonlines.open(args.output_file + "_dev.jsonl", "w") as writer:
        writer.write_all(dev_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inst_file",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )

    parser.add_argument(
        "--only_negative",
        action="store_true"
    )

    parser.add_argument(
        "--kd",
        action="store_true"
    )
    parser.add_argument(
        "--ignore_tasks",
        type=str,
        nargs="+"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--sample_dataset_num", type=int, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to passages (.tsv file)")
    parser.add_argument("--task_names", type=str, default=None,
                        help="task names filter", nargs="+")
    parser.add_argument("--prompts", type=str, default=None,
                        help="prompt", nargs="+")
    parser.add_argument("--per_gpu_batch_size", type=int,
                        default=64, help="Batch size for question encoding")
    parser.add_argument("--n_docs", type=int, default=100,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--instruction_unfollowing_file", type=str, default=None,
                        help="instruction unfollowing file")
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true",
                        help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int,
                        default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--start_idx", type=int, default=None,)
    parser.add_argument("--end_idx", type=int, default=None,)
    parser.add_argument("--num_positive_paragraphs", type=int, default=2,)
    parser.add_argument("--num_negative_paragraphs", type=int, default=7,)
    parser.add_argument("--num_hard_negative_paragraphs", type=int, default=3,)
    parser.add_argument(
        "--num_instructions_unfollowing_negatives", type=int, default=1,)
    args = parser.parse_args()
    main(args)
