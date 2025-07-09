from transformers import AutoTokenizer
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split


class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        news = self.data[index]

        text = ""
        if news["publish_time"]:
            text = news["publish_time"] + ","
        if news["title"]:
            text += news["title"]
        if news["content"]:
            text += news["content"]

        text = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        s = ""
        for i in range(len(news["backgrounds"])):
            if i > 0:
                s += ";"
            s += news["backgrounds"][i]
        s += "."

        backgrounds = self.tokenizer(
            s,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        s = ""
        for i in range(len(news["issues"])):
            if i > 0:
                s += ";"
            s += news["issues"][i]
        s += "."

        issues = self.tokenizer(
            s,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label = torch.tensor(news["label"], dtype=torch.float)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "backgrounds_input_ids": backgrounds["input_ids"].squeeze(0),
            "backgrounds_attention_mask": backgrounds["attention_mask"].squeeze(0),
            "issues_input_ids": issues["input_ids"].squeeze(0),
            "issues_attention_mask": issues["attention_mask"].squeeze(0),
            "label": label,
        }


# load training data
def load_training_data(pretrained_model_file,seed=3759, train_ratio=0.7, val_ratio=0.2, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_file, local_files_only=True)
    with open("./dataset/training.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    dataset = NewsDataset(data, tokenizer)

    torch.manual_seed(seed)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for batch in test_dataloader:
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #     backgrounds_input_ids = batch["backgrounds_input_ids"]
    #     backgrounds_attention_mask = batch["backgrounds_attention_mask"]
    #     issues_input_ids = batch["issues_input_ids"]
    #     issues_attention_mask = batch["issues_attention_mask"]
    #     labels = batch["label"]

    #     print(f"input_ids shape: {input_ids.shape}")
    #     print(f"attention_mask shape: {attention_mask.shape}")
    #     print(f"backgrounds_input_ids shape: {backgrounds_input_ids.shape}")
    #     print(f"backgrounds_attention_mask shape: {backgrounds_attention_mask.shape}")
    #     print(f"issues_input_ids shape: {issues_input_ids.shape}")
    #     print(f"issues_attention_mask shape: {issues_attention_mask.shape}")
    #     print(f"labels shape: {labels.shape}")

    #     break

    return train_dataloader, val_dataloader, test_dataloader

# load test data
def load_test_data(pretrained_model_file,batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_file, local_files_only=True)
    with open("./dataset/test.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    dataset = NewsDataset(data, tokenizer)

    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return dataloader

