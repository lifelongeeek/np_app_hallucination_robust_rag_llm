import os
import torch
import argparse
import numpy as np
import pandas as pd

from loguru import logger
from collections import defaultdict
from prompts import SINGLE_TOKEN_BASELINE_PROMPT
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="google/gemma-1.1-7b-it")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--token", type=str, help="huggingface access token for gated models", default=None)
    parser.add_argument("--device", type=str, help="gpu to use", default="cuda:0")
    parser.add_argument("--print_every_response", action="store_true")

    args = parser.parse_args(args)
    return args


def get_model(args):
    lora_weights = args.model_path
    config = PeftConfig.from_pretrained(lora_weights)
    base_model_id = config.base_model_name_or_path

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map=args.device,
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model.eval()


def get_response(model, tokenizer, final_prompt):
    device = model.device
    inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
    logits = model(**inputs).logits

    labels = [" U", " P", " A"]
    tokens = tokenizer(labels)["input_ids"]
    # BOS token handling
    tokens = list(map(lambda x: x[-1], tokens))

    probs = np.array([logits[0, -1][token].item() for token in tokens])
    max_index = np.argmax(probs)

    choices = ["[Unanswerable]", "[Partially answerable]", "[Answerable]"]
    return choices[max_index]


def main(args):
    tokenizer, model = get_model(args)

    evalset = load_dataset("json", data_files=args.dataset_path, split="train")

    label_dict = defaultdict(list)
    pred_dict = defaultdict(list)

    id_mapping = {"[Unanswerable]": 0, "[Partially answerable]": 1, "[Answerable]": 2}

    for data in tqdm(evalset):
        dataset = data["dataset"]
        query = data["query"]
        apis = data["api_string"]
        label = data["label"]

        final_prompt = SINGLE_TOKEN_BASELINE_PROMPT.format(query=query, apis=apis)
        response = get_response(model, tokenizer, final_prompt)

        label_dict[dataset].append(id_mapping[label])
        pred_dict[dataset].append(id_mapping[response])

        if args.print_every_response:
            print(label, response)

    datasets = list(label_dict.keys())
    total_preds, total_labels, save_rows = [], [], []

    for dataset in datasets:
        label_list = label_dict[dataset]
        pred_list = pred_dict[dataset]

        total_labels.extend(label_list)
        total_preds.extend(pred_list)

        precision = precision_score(label_list, pred_list, average="macro", zero_division=0.0)
        recall = recall_score(label_list, pred_list, average="macro", zero_division=0.0)
        macro_f1 = f1_score(label_list, pred_list, average="macro", zero_division=0.0)
        micro_f1 = f1_score(label_list, pred_list, average="micro", zero_division=0.0)

        print(f"{dataset}: precision: {precision:0.3f}, recall: {recall:0.3f}, macro_f1: {macro_f1:0.3f}, micro_f1: {micro_f1:0.3f}")
        save_rows.append([dataset, precision, recall, macro_f1, micro_f1])

    precision = precision_score(total_labels, total_preds, average="macro", zero_division=0.0)
    recall = recall_score(total_labels, total_preds, average="macro", zero_division=0.0)
    macro_f1 = f1_score(total_labels, total_preds, average="macro", zero_division=0.0)
    micro_f1 = f1_score(total_labels, total_preds, average="micro", zero_division=0.0)
    save_rows.append(["Total", precision, recall, macro_f1, micro_f1])

    print(f"Total: precision: {precision:0.3f}, recall: {recall:0.3f}, macro_f1: {macro_f1:0.3f}, micro_f1: {micro_f1:0.3f}")

    df = pd.DataFrame(save_rows, columns=["dataset", "precision", "recall", "macro_f1", "micro_f1"])
    save_path = os.path.join(args.model_path, "result.csv")
    df.to_csv(save_path, index=False)
    print("Result saved at", save_path)


if __name__ == "__main__":
    args = parse_args(None)

    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    main(args)
