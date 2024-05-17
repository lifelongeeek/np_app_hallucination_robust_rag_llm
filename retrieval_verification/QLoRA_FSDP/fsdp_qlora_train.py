import os
import random
import torch
import numpy as np
import wandb
import pickle
import datetime
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig
from loguru import logger
from trl import SFTTrainer
from prompts import BASELINE_PROMPT, SINGLE_TOKEN_BASELINE_PROMPT


def set_custom_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={"help": "Path to the dataset"},
    )
    model_id: str = field(default=None, metadata={"help": "Model ID to use for SFT training"})
    max_seq_length: int = field(default=4096, metadata={"help": "The maximum sequence length for SFT Trainer"})
    train_task: str = field(default=None, metadata={"help": "Token to use(raw label or single token)"})


@dataclass
class LogArguments:
    exp_name: str = field(default=None, metadata={"help": "Path to the dataset"})
    output_base_dir: str = field(default=None, metadata={"help": "Base directory for model saving"})


def get_dataset(tokenizer, train_task):
    def generate_and_tokenize_prompt(data_point):
        if train_task == "raw":
            user_prompt = BASELINE_PROMPT.format(query=data_point["query"], apis=data_point["apis"])
            full_prompt = f"{user_prompt}[{data_point['pseudo_label']}]{tokenizer.eos_token}"
        elif train_task == "single_token":
            label_to_short = {"Answerable": "A", "Partially answerable": "P", "Unanswerable": "U"}
            user_prompt = SINGLE_TOKEN_BASELINE_PROMPT.format(query=data_point["query"], apis=data_point["apis"])
            full_prompt = f"{user_prompt}{label_to_short[data_point['pseudo_label']]}"
        else:
            print("train_task only support ['raw', 'single_token']")
            raise NotImplementedError

        tokenized_user_prompt = tokenizer(user_prompt, truncation=True, padding=True)
        user_prompt_len = len(tokenized_user_prompt["input_ids"]) - 1

        tokenized_full_prompt = tokenizer(full_prompt, truncation=True, padding=True)
        tokenized_full_prompt["labels"] = tokenized_full_prompt["input_ids"].copy()

        tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        tokenized_full_prompt["text"] = full_prompt

        return tokenized_full_prompt

    dataset_path = os.path.join(os.getcwd(), f"dataset_4k_{train_task}.pkl")
    if os.path.isfile(dataset_path):
        with open(dataset_path, "rb") as f:
            trainset, validset, plav_eval = pickle.load(f)
    else:
        dataset_name = "PLAV_trainset_0508_apu.jsonl"
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.1)

        plav_eval = load_dataset("json", data_files="plav_for_eval.jsonl", split="train")

        trainset = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
        validset = dataset["test"].map(generate_and_tokenize_prompt)
        plav_eval = plav_eval.map(generate_and_tokenize_prompt)

        trainset = [d for d in trainset if len(d["input_ids"]) < 4096]
        validset = [d for d in validset if len(d["input_ids"]) < 4096]
        plav_eval = [d for d in plav_eval if len(d["input_ids"]) < 4096]

        with open(dataset_path, "wb") as f:
            pickle.dump([trainset, validset, plav_eval], f)
        print(f"Dataset saved at {dataset_path}")

    return trainset, validset, plav_eval


def training_function(script_args, training_args):
    ################
    # TOKENIZER
    ################
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ################
    # DATASET
    ################
    train_dataset, test_dataset, plav_eval = get_dataset(tokenizer, script_args.train_task)

    ################
    # MODEL
    ################
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation="sdpa",  # use sdpa, alternatively use "flash_attention_2"
        torch_dtype=quant_storage_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset={"stblib": test_dataset, "plav": plav_eval},
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = TrlParser((LogArguments, ScriptArguments, TrainingArguments))
    log_args, script_args, training_args = parser.parse_args_and_config()

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    current_datetime = datetime.datetime.now()
    hours_to_add = datetime.timedelta(hours=9)
    result_datetime = current_datetime + hours_to_add
    formatted_datetime = result_datetime.strftime("%m-%d_%H:%M:%S")

    exp_postfix = f"bs{training_args.per_device_train_batch_size}_acc{training_args.gradient_accumulation_steps}_gpu{world_size}"
    exp_name = f"{log_args.exp_name}_{script_args.train_task}_{exp_postfix}_{formatted_datetime}"
    output_dir = os.path.join(log_args.output_base_dir, exp_name)
    training_args.output_dir = output_dir

    if global_rank == 0:
        wandb.init(project="plav-vf-model-cls", name=exp_name)

        logger.info("log parameters")
        for k, v in vars(log_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

        logger.info("training parameters")
        for k, v in vars(script_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

        logger.info("training parameters")
        for k, v in vars(training_args).items():
            logger.info(f"{k:30} {v}")
        logger.info("*" * 40)

    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
    set_custom_seed(training_args.seed)

    # launch training
    training_function(script_args, training_args)
