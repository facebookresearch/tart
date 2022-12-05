# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
from turtle import update
import torch
import logging
import json
import numpy as np
import os
import copy

import src.slurm
import src.contriever
import src.beir_utils
import src.utils
import src.dist_utils
import src.contriever

logger = logging.getLogger(__name__)


def main(args):

    src.slurm.init_distributed_mode(args)
    src.slurm.init_signal_handler()

    os.makedirs(args.output_dir, exist_ok=True)

    logger = src.utils.init_logger(args)

    model, tokenizer, _ = src.contriever.load_retriever(
        args.model_name_or_path)
    if args.bi_encoder is True:
        if args.ckpt_path is not None:
            state_dict = torch.load(args.ckpt_path)["model"]
            query_encoder = copy.deepcopy(model)
            doc_encoder = copy.deepcopy(model)
            query_encoder_dic = {}
            doc_encoder_dic = {}
            for name, param in state_dict.items():
                print(name)
                if "q_encoder" in name:
                    if "encoder" in name:
                        orig_name = name.replace(
                            "q_encoder.encoder", "encoder")
                    if "embeddings" in name:
                        orig_name = name.replace(
                            "q_encoder.embeddings", "embeddings")
                    query_encoder_dic[orig_name] = param
                    print(orig_name)
                if "p_encoder" in name:
                    if "encoder" in name:
                        orig_name = name.replace(
                            "p_encoder.encoder", "encoder")
                    if "embeddings" in name:
                        orig_name = name.replace(
                            "p_encoder.embeddings", "embeddings")
                    print(orig_name)
                    doc_encoder_dic[orig_name] = param

            # print(query_encoder_dic.keys())
            # print(doc_encoder_dic.keys())
            query_encoder.load_state_dict(query_encoder_dic)
            # doc_encoder.load_state_dict(doc_encoder_dic)

            query_encoder = query_encoder.cuda()
            query_encoder.eval()

            doc_encoder = doc_encoder.cuda()
            doc_encoder.eval()
        else:
            print("for biencoder model, you have to preload the fine-tuned checkpoints.")
            raise NotImplementedError()

    else:
        if args.ckpt_path is not None:
            print("loading model")
            state_dict = torch.load(args.ckpt_path)["model"]

            new_dict = {}
            # state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items() if "encoder_q." in k}
            print(state_dict.keys())
            # print(dict(model.named_parameters()).keys())

            for name, param in model.named_parameters():
                if name in state_dict:
                    new_dict[name] = state_dict[name]
                    print("updated")
                    print(name)
                else:
                    new_dict[name] = param

            # print(new_dict.keys())
            # print(model.keys())
            # assert model.keys() == new_dict.keys()

            model.load_state_dict(new_dict, strict=False)

        model = model.cuda()
        model.eval()
        query_encoder = model
        doc_encoder = model

    logger.info("Start indexing")

    if args.multiple_prompts is not None:
        metrics = src.beir_utils.evaluate_model_multiple(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            dataset=args.dataset,
            batch_size=args.per_gpu_batch_size,
            norm_query=args.norm_query,
            norm_doc=args.norm_doc,
            is_main=src.dist_utils.is_main(),
            split="dev" if args.dataset == "msmarco" else "test",
            score_function=args.score_function,
            beir_dir=args.beir_dir,
            save_results_path=args.save_results_path,
            lower_case=args.lower_case,
            normalize_text=args.normalize_text,
            prompt=args.prompt,
            multiple_prompts=args.multiple_prompts
        )
    else:
        metrics = src.beir_utils.evaluate_model(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            dataset=args.dataset,
            batch_size=args.per_gpu_batch_size,
            norm_query=args.norm_query,
            norm_doc=args.norm_doc,
            is_main=src.dist_utils.is_main(),
            split="dev" if args.dataset == "msmarco" else "test",
            score_function=args.score_function,
            beir_dir=args.beir_dir,
            save_results_path=args.save_results_path,
            lower_case=args.lower_case,
            normalize_text=args.normalize_text,
            prompt=args.prompt,
            emb_load_path=args.emb_load_path,
            emb_save_path=args.emb_save_path
        )

    if src.dist_utils.is_main():
        for key, value in metrics.items():
            logger.info(f"{args.dataset} : {key}: {value:.1f}")

    print("saving results")
    if os.path.exists(os.path.join(args.output_dir, "{0}_{1}_results.json".format(args.dataset, args.model_id))) is True:
        results_log = json.load(open(os.path.join(
            args.output_dir, "{0}_{1}_results.json".format(args.dataset, args.model_id))))
    else:
        results_log = {}
    results_log.setdefault(args.prompt, {})
    results_log[args.prompt] = metrics

    with open(os.path.join(args.output_dir, "{0}_{1}_results.json".format(args.dataset, args.model_id)), "w") as outfile:
        json.dump(results_log, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str,
                        help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./",
                        help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int,
                        default=512, help="Maximum text length")
    parser.add_argument("--emb_load_path", type=str, default=None,
                        help="path to load already computed embeddings.", nargs="+")
    parser.add_argument("--emb_save_path", type=str, default=None,
                        help="path to save already computed embeddings.")

    parser.add_argument("--per_gpu_batch_size", default=128,
                        type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str,
                        default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str,
                        help="Model name or path")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--norm_query", action="store_true",
                        help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true",
                        help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true",
                        help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_results_path", type=str,
                        default=None, help="Path to save result object")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
    parser.add_argument("--ckpt_path", type=str, help="Model name or path")
    parser.add_argument("--bi_encoder",  action="store_true", )
    parser.add_argument(
        "--prompt", type=str, default=None, help="instructional prompt."
    )
    parser.add_argument(
        "--multiple_prompts", type=str, nargs='+'
    )
    parser.add_argument(
        "--model_id", type=str, default=None, help="for logging"
    )
    args, _ = parser.parse_known_args()
    main(args)
