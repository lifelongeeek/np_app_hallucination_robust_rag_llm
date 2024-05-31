import os
import torch
import pickle
import pandas as pd
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score


def get_model(lora_weights_path):
    config = PeftConfig.from_pretrained(lora_weights_path)
    base_model_id = config.base_model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        config=config,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model.eval()


if __name__ == "__main__":
    dataset_path = "<pickle_file_path>"
    with open(dataset_path, "rb") as f:
        trainset, validset, plavset = pickle.load(f)

    lora_weights_path = "<lora_weights_path>"
    tokenizer, model = get_model(lora_weights_path)

    label_tokens = tokenizer([" U P A"])["input_ids"][0][1:]

    def get_eval(dataset):
        total_losses = 0
        total_preds, total_labels = [], []

        for t in tqdm(dataset):
            # tokens = tokenizer(t["text"], return_tensors="pt").to("cuda:0")
            # import pdb

            # pdb.set_trace()
            tokens = torch.Tensor(t["input_ids"]).long().unsqueeze(dim=0).to("cuda:0")
            labels = torch.Tensor(t["labels"]).long().unsqueeze(dim=0)
            output = model(tokens, labels=labels)
            tokens.cpu().detach()
            total_losses += output.loss.item()

            last_logits = output.logits[0, -2]
            label_logits = torch.Tensor([last_logits[t].item() for t in label_tokens])
            pred = label_logits.argmax().item()
            label = label_tokens.index(labels[0][-1])

            total_preds.append(pred)
            total_labels.append(label)

        ave_losses = total_losses / len(dataset)
        precision = precision_score(total_labels, total_preds, average="macro", zero_division=0.0)
        recall = recall_score(total_labels, total_preds, average="macro", zero_division=0.0)
        macro_f1 = f1_score(total_labels, total_preds, average="macro", zero_division=0.0)
        micro_f1 = f1_score(total_labels, total_preds, average="micro", zero_division=0.0)

        print(f"loss: {ave_losses:0.3f}, precision: {precision:0.3f}, recall: {recall:0.3f}, macro_f1: {macro_f1:0.3f}, micro_f1: {micro_f1:0.3f}")
        return [ave_losses, precision, recall, macro_f1, micro_f1]

    print("Evaluating trainset ...")
    train_subset = trainset.train_test_split(test_size=0.1)["test"]
    row1 = get_eval(train_subset)
    row1 = ["trainset"] + row1
    print("Evaluating validset ...")
    row2 = get_eval(validset)
    row2 = ["validset"] + row2
    print("Evaluating plavset ...")
    row3 = get_eval(plavset)
    row3 = ["plavset"] + row3

    df = pd.DataFrame([row1, row2, row3], columns=["dataset", "loss", "precision", "recall", "macro_f1", "micro_f1"])
    save_path = os.path.join(lora_weights_path, "train_eval_result.csv")
    df.to_csv(save_path, index=False)
    print(f"Result saved at {save_path}")
