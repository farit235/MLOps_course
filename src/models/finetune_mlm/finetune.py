import math
import typing as t

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)


class TokenizedSentsDataset(Dataset):
    def __init__(
        self,
        texts: t.List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        tokenization_batch_size: int = 50,
        verbose: int = True,
    ):
        self.tokenized_texts = []
        range_num_batches = range(math.ceil(len(texts) / tokenization_batch_size))
        if verbose:
            range_num_batches = tqdm(range_num_batches)
        for batch_id in range_num_batches:
            batch = texts[
                (tokenization_batch_size * batch_id) : (
                    tokenization_batch_size * (batch_id + 1)
                )
            ]
            self.tokenized_texts += tokenizer(
                batch, max_length=max_length, truncation=True
            )["input_ids"]
        # sort by length to optimize computation
        self.tokenized_texts = sorted(self.tokenized_texts, key=lambda x: -len(x))

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx]


# toDo add  evaluate after each epoch
def finetune(
    initial_name_or_path: str,
    save_path: str,
    train_sents: t.List[str],
    batch_size: int = 50,
    lr: float = 1e-5,
    mlm_probability: float = 0.15,
    num_epochs: int = 10,
    shedular_name: str = "linear",
    verbose: bool = True,
) -> None:
    # toDo fix config instead of ignoring mismatched
    model = BertForMaskedLM.from_pretrained(
        initial_name_or_path, ignore_mismatched_sizes=True
    )
    tokenizer = BertTokenizer.from_pretrained(initial_name_or_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    train_dataset = TokenizedSentsDataset(train_sents, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name=shedular_name,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if verbose:
        progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {key: tensor.to(device) for key, tensor in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if verbose:
                progress_bar.update(1)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    return
