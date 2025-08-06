import os.path as osp
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from sentence_transformers import SentenceTransformer
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from src.benchmarks import get_qa_dataset, get_semistructured_data
from src.models import get_model
from src.tools.args import merge_args, load_args
import logging
import time
from transformers import AutoTokenizer,AutoModelForCausalLM
from bisect import bisect_left

def get_rank(dic, node_id):
    sorted_rank = sorted(dic.values(), reverse=True)
    given_scores = dic[node_id]
    index = bisect_left(sorted_rank, given_scores)
    return index + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="amazon", choices=['amazon', 'primekg', 'mag'])
    parser.add_argument(
        "--model", default="VSS", choices=["VSS", "MultiVSS", "LLMReranker", "QSS"]
    )
    parser.add_argument("--split", default="test")

    # can eval on a subset only
    parser.add_argument("--test_ratio", type=float, default=1.0)

    # for multivss
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--multi_vss_topk", type=int, default=None)
    parser.add_argument("--aggregate", type=str, default="max")

    # for vss, multivss, and llm reranker
    parser.add_argument("--emb_model", type=str, default="gte")

    # for llm reranker
    parser.add_argument("--llm_model", type=str, default="./llm_model/Meta-Llama-3-8B-Instruct",
                        help='the LLM to rerank candidates.')
    parser.add_argument("--llm_topk", type=int, default=20)
    parser.add_argument("--max_retry", type=int, default=3)

    # path
    parser.add_argument("--emb_dir", type=str, default='emb/')
    parser.add_argument("--output_dir", type=str, default='result/')

    # save prediction
    parser.add_argument("--save_pred", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_args = load_args(
        json.load(open("config/default_args.json", "r"))[args.dataset]
    )
    args = merge_args(args, default_args)    

    args.query_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "query")
    args.node_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "doc")
    args.chunk_emb_dir = osp.join(args.emb_dir, args.dataset, args.emb_model, "chunk")
    surfix = args.emb_model
    output_dir = osp.join(args.output_dir, "eval", args.dataset, args.model, surfix)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    os.makedirs(f"{output_dir}/logs/", exist_ok=True)
    fh = logging.FileHandler(f"{output_dir}/logs/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.query_emb_dir, exist_ok=True)
    os.makedirs(args.chunk_emb_dir, exist_ok=True)
    os.makedirs(args.node_emb_dir, exist_ok=True)
    json.dump(vars(args), open(osp.join(output_dir, "args.json"), "w"), indent=4)

    eval_csv_path = osp.join(output_dir, f"eval_results_{args.split}.csv")
    final_eval_path = (
        osp.join(output_dir, f"eval_metrics_{args.split}.json")
        if args.test_ratio == 1.0
        else osp.join(output_dir, f"eval_metrics_{args.split}_{args.test_ratio}.json")
    )

    kb = get_semistructured_data(args.dataset)
    qa_dataset = get_qa_dataset(args.dataset)
    
    llm_model_pt = None
    tokenizer = None
    if "Reranker" in args.model or args.model == "QSS":
        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=True)
        # llama 如果要训练，数据类型需要是bfloat16
        llm_model_pt = AutoModelForCausalLM.from_pretrained(args.llm_model, device_map="auto", torch_dtype=torch.bfloat16)

    model = get_model(args=args, kb=kb, llm_model_pt=llm_model_pt, tokenizer=tokenizer)

    split_idx = qa_dataset.get_idx_split(test_ratio=args.test_ratio)

    eval_metrics = [
        "mrr",
        "map",
        "rprecision",
        "recall@5",
        "recall@10",
        "recall@20",
        "recall@50",
        "recall@100",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "hit@20",
        "hit@50",
        "hit@100"
    ]
    eval_csv = pd.DataFrame(columns=["idx", "query_id", "pred_rank"] + eval_metrics)

    existing_idx = []
    if osp.exists(eval_csv_path):
        eval_csv = pd.read_csv(eval_csv_path)
        existing_idx = eval_csv["idx"].tolist()
        logger.info(f"load eval csv from {eval_csv_path}")

    indices = split_idx[args.split].tolist()
    top_k_nodes = {}
    for idx in tqdm(indices):
        if idx in existing_idx:
            continue
        query, query_id, answer_ids, meta_info = qa_dataset[idx]
        
        if args.model == "QSS":
            pred_dict, chosen_node, pred_dict_ori, init_score, text, origin_text = model.forward(query, query_id)
        elif args.model == "VSS":
            pred_dict, _, top_k_node = model.forward(query, query_id)
            top_k_nodes[query_id] = top_k_node
        elif args.model == "MultiVSS":
            pred_dict, top_k_node = model.forward(query, query_id, answer_ids)
            top_k_nodes[query_id] = top_k_node
        elif args.model == "LLMReranker":
            pred_dict, pred_dict_ori = model.forward(query, query_id)
        else:
            pred_dict = model.forward(query, query_id)

        answer_ids = torch.LongTensor(answer_ids)
        result = model.evaluate(pred_dict, answer_ids, metrics=eval_metrics)

        result["idx"], result["query_id"] = idx, query_id
        result["pred_rank"] = torch.LongTensor(list(pred_dict.keys()))[
            torch.argsort(torch.tensor(list(pred_dict.values())), descending=True)[
                :1000
            ]
        ].tolist()

        eval_csv = pd.concat([eval_csv, pd.DataFrame([result])], ignore_index=True)
        for metric in eval_metrics:
            logging.info(f'this question {metric}: {result[metric]}')

        if args.save_pred:
            eval_csv.to_csv(eval_csv_path, index=False)
        for metric in eval_metrics:
            logger.info(
                f"total {metric}: {np.mean(eval_csv[eval_csv['idx'].isin(indices)][metric])}"
            )
        
    if args.save_pred:
        eval_csv.to_csv(eval_csv_path, index=False)
    final_metrics = (
        eval_csv[eval_csv["idx"].isin(indices)][eval_metrics].mean().to_dict()
    )
    json.dump(final_metrics, open(final_eval_path, "w"), indent=4)