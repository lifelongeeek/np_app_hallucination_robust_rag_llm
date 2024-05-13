import os
import pickle
import torch
import argparse
import numpy as np
import wandb

from accelerate import Accelerator
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

from datasets import load_dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from prompts import BASELINE_PROMPT
from loguru import logger

os.environ["TRANSFORMERS_CACHE"] = "/data/jykim/cache"


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="google/gemma-1.1-7b-it")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="gemma/exp6")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--token", type=str, help="huggingface access token for gated models")

    args = parser.parse_args(args)
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_model(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        token=args.token,
        cache_dir="/data/jykim/cache",
    )
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        add_eos_token=True,
        use_fast=True,
        token=args.token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def get_dataset(tokenizer):
    def generate_and_tokenize_prompt(data_point):
        user_prompt = BASELINE_PROMPT.format(query=data_point["query"], apis=data_point["apis"])
        if "pseudo_label" in data_point:
            full_prompt = f"{user_prompt}[{data_point['pseudo_label']}]{tokenizer.eos_token}"
        else:
            full_prompt = user_prompt

        tokenized_user_prompt = tokenizer(user_prompt, truncation=True, padding=True)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - 1

        tokenized_full_prompt = tokenizer(full_prompt, truncation=True, padding=True)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

        return tokenized_full_prompt

    dataset_path = "dataset.pkl"
    if os.path.isfile(dataset_path):
        with open(dataset_path, "rb") as f:
            trainset, validset = pickle.load(f)
    else:
        dataset_name = "PLAV_trainset_0508_apu.jsonl"
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.05)

        trainset = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
        validset = dataset["test"].map(generate_and_tokenize_prompt)

        with open(dataset_path, "wb") as f:
            pickle.dump([trainset, validset], f)

    return trainset, validset


def main(args):
    wandb.init(project="plav-vf-model-cls", name=args.exp_name)
    set_seed(42)
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    tokenizer, model = get_model(args)
    trainset, validset = get_dataset(tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=True,
        save_total_limit=5,
        logging_steps=5,
        output_dir=args.save_dir,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=args.eval_step,
        save_steps=args.eval_step,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        warmup_ratio=0.3,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    accelerator = Accelerator()
    trainer = accelerator.prepare(
        Trainer(
            model=model,
            train_dataset=trainset,
            eval_dataset=validset,
            args=training_args,
            data_collator=data_collator,
        )
    )

    model.config.use_cache = False

    trainer.train()


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
