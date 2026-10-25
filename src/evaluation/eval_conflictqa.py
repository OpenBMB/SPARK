#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parametric-Faithful evaluation on ConFiQA.

Input:  cf_context  (counterfactual context)
GT:     orig_answer + orig_alias  (factually correct / parametric answer)

Tests whether the model can REJECT counterfactual context and preserve
its correct parametric knowledge.
"""

import os
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter

os.environ["VLLM_USE_V1"] = "0"

import argparse
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--model", type=str, required=True)
_arg_parser.add_argument("--tokenizer", type=str, default=None,
    help="Tokenizer dir (defaults to --model)")
_arg_parser.add_argument("--cuda", type=str, default="6")
_arg_parser.add_argument("--gpu_mem", type=float, default=0.4)
_args, _ = _arg_parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = _args.cuda

MODEL_WEIGHTS_DIR = _args.model
MODEL_CODE_DIR    = _args.tokenizer if _args.tokenizer else _args.model

TEST_DATA_DIR = Path("/your/path/to/conflictqa") # change to real path

MAX_RESPONSE_TOKENS = 1024
TEMPERATURE         = 0.6
TOP_P               = 0.95
MAX_INPUT_LENGTH    = 7000

SYSTEM_MESSAGE = (
    "You are a helpful assistant that answers questions based on the provided background information. "
    "The background information may be incorrect, so you should judge whether to believe yourself or to believe the background information."
)

PROMPT_TEMPLATE = (
    "Background:\n{context}\n\n"
    "Task Instruction:\n"
    "Answer the question."
    "Provide your final answer in <answer> </answer> tags, "
    "for example <answer>Petri Alanko</answer>.\n\n"
    "Q: {question}\n\n"
    "A: "
)

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"\u2018", u"\u2019", u"\u00b4", u"\u0060"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)
    def replace_underscore(text):
        return text.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(replace_underscore(s.lower())))).strip()


def extract_answer(solution_str: str) -> Optional[str]:
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def acc_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) in normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gt_tokens) if gt_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths) -> float:
    if not isinstance(ground_truths, list):
        ground_truths = [ground_truths]
    return max(
        metric_fn(normalize_answer(prediction), normalize_answer(gt))
        for gt in ground_truths
    )


def load_and_preprocess(json_path: Path, tokenizer: AutoTokenizer) -> Dataset:
    """
    Input:  cf_context (counterfactual context)
    GT:     orig_answer + orig_alias (correct parametric answer)
    """
    raw = load_dataset("json", data_files=str(json_path), split="train")

    def preprocess(example):
        question = example["question"]
        context = example["cf_context"]

        prefix = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )},
        ]
        input_ids = tokenizer.apply_chat_template(
            prefix, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )

        gt_list = [example["orig_answer"]]
        orig_alias = example.get("orig_alias", [])
        if isinstance(orig_alias, list):
            gt_list.extend(orig_alias)

        return {
            "input_ids":      input_ids,
            "question":       question,
            "orig_answer":    gt_list,
            "cf_answer":      example.get("cf_answer", ""),
        }

    ds = raw.map(preprocess, num_proc=4, remove_columns=raw.column_names)
    before = len(ds)
    ds = ds.filter(lambda x: len(x["input_ids"]) <= MAX_INPUT_LENGTH, num_proc=4)
    after = len(ds)
    if before != after:
        print(f"  [filter] {before} -> {after} samples (removed {before - after} too-long)")
    return ds


def evaluate_dataset(
    dataset_path: Path,
    inference_engine: LLM,
    tokenizer: AutoTokenizer,
    eos_token_id: int,
    eos_token: str,
) -> Dict[str, Any]:
    print(f"\n{'─'*60}")
    print(f"  Evaluating (Param-Faithful): {dataset_path.name}")
    print(f"{'─'*60}")

    ds = load_and_preprocess(dataset_path, tokenizer)
    n = len(ds)
    print(f"  Samples after filtering: {n}")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_RESPONSE_TOKENS,
        n=1,
        detokenize=False,
        stop_token_ids=[eos_token_id],
    )

    generations = inference_engine.generate(
        [{"prompt_token_ids": ids} for ids in ds["input_ids"]],
        sampling_params=sampling_params,
    )

    total_em = total_acc = total_f1 = 0.0
    # also track how many answers matched cf_answer (followed counterfactual)
    total_cf_match = 0.0

    for i, sample in enumerate(ds):
        response_token_ids = generations[i].outputs[0].token_ids
        response = tokenizer.decode(response_token_ids, skip_special_tokens=False)

        if response.endswith(eos_token):
            response = response[: -len(eos_token)]

        prediction = extract_answer(response) or ""
        ground_truth = sample["orig_answer"]

        total_em  += metric_max_over_ground_truths(exact_match_score, prediction, ground_truth)
        total_acc += metric_max_over_ground_truths(acc_score, prediction, ground_truth)
        total_f1  += metric_max_over_ground_truths(f1_score, prediction, ground_truth)

        cf_ans = sample["cf_answer"]
        if cf_ans:
            total_cf_match += exact_match_score(normalize_answer(prediction), normalize_answer(cf_ans))

    avg_em  = 100.0 * total_em  / n if n else 0.0
    avg_acc = 100.0 * total_acc / n if n else 0.0
    avg_f1  = 100.0 * total_f1  / n if n else 0.0
    avg_cf  = 100.0 * total_cf_match / n if n else 0.0

    result = {
        "dataset": dataset_path.stem,
        "n":       n,
        "EM (%)":  round(avg_em, 2),
        "ACC (%)": round(avg_acc, 2),
        "F1 (%)":  round(avg_f1, 2),
        "CF-Follow (%)": round(avg_cf, 2),
    }

    print(f"  Param-Faithful EM  : {avg_em:.2f}%")
    print(f"  Param-Faithful ACC : {avg_acc:.2f}%")
    print(f"  Param-Faithful F1  : {avg_f1:.2f}%")
    print(f"  CF-Follow Rate     : {avg_cf:.2f}%  (followed counterfactual context)")

    return result


def main():
    hf_ckpt = MODEL_WEIGHTS_DIR

    print("=" * 60)
    print(f"  Parametric-Faithful Evaluation on ConFiQA")
    print(f"  Model path: {hf_ckpt}")
    print(f"  Input: cf_context | GT: orig_answer")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CODE_DIR, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    eos_token    = tokenizer.convert_ids_to_tokens(eos_token_id)
    print(f"EOS token: {repr(eos_token)}  (id={eos_token_id})")

    print("\nInitialising vLLM engine ...")
    inference_engine = LLM(
        model=str(hf_ckpt),
        tokenizer=MODEL_CODE_DIR,
        skip_tokenizer_init=False,
        gpu_memory_utilization=_args.gpu_mem,
        enable_prefix_caching=True,
        dtype=torch.bfloat16,
        max_model_len=8192,
        trust_remote_code=True,
    )
    print("vLLM engine ready.\n")

    test_files = sorted(TEST_DATA_DIR.glob("ConFiQA-*.json"))
    if not test_files:
        raise RuntimeError(f"No ConFiQA-*.json files found in {TEST_DATA_DIR}")

    print(f"Found {len(test_files)} test datasets:")
    for f in test_files:
        print(f"  {f.name}")

    all_results: List[Dict[str, Any]] = []
    for test_file in test_files:
        result = evaluate_dataset(
            dataset_path=test_file,
            inference_engine=inference_engine,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            eos_token=eos_token,
        )
        all_results.append(result)

    print("\n" + "=" * 80)
    print(f"  SUMMARY — Parametric-Faithful  (Model: {Path(hf_ckpt).name})")
    print("=" * 80)
    header = f"{'Dataset':<30} {'N':>6} {'EM%':>8} {'ACC%':>8} {'F1%':>8} {'CF-Fol%':>8}"
    print(header)
    print("─" * 80)
    for r in all_results:
        print(f"  {r['dataset']:<28} {r['n']:>6}"
              f" {r['EM (%)']:>8.2f} {r['ACC (%)']:>8.2f} {r['F1 (%)']:>8.2f}"
              f" {r['CF-Follow (%)']:>8.2f}")

    total_n  = sum(r["n"] for r in all_results)
    avg_em   = sum(r["EM (%)"]  * r["n"] for r in all_results) / total_n if total_n else 0.0
    avg_acc  = sum(r["ACC (%)"] * r["n"] for r in all_results) / total_n if total_n else 0.0
    avg_f1   = sum(r["F1 (%)"]  * r["n"] for r in all_results) / total_n if total_n else 0.0
    avg_cf   = sum(r["CF-Follow (%)"] * r["n"] for r in all_results) / total_n if total_n else 0.0

    print("─" * 80)
    print(f"  {'MICRO-AVG':<28} {total_n:>6}"
          f" {avg_em:>8.2f} {avg_acc:>8.2f} {avg_f1:>8.2f} {avg_cf:>8.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
