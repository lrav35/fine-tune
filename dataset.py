from typing import Dict, Sequence
from dataclasses import dataclass
import torch
import math
import os
import transformers
import datasets
from torch.utils.data import Dataset
import copy

def fmt_prompt(prompt):
    return f"### Instructions:\n Can you please translate this phrase or word to french? \n {prompt}\n\n### Response:\n Yes of course! Here is a french translation of that phrase: \n"

def _tokenize(
        strings: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """tokenize examples"""
    tokenized_strings = [
        tokenizer(
            example,
            return_tensors='pt',
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) 
        for example in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_strings]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_strings
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
        samples: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
    """Preprocess data for training by tokenizing"""
    sources = [f"{fmt_prompt(sources)}" for sources in samples["input"]]
    targets = [f"{translation}{tokenizer.eos_token}" for translation in samples["output"]]
    complete_examples = [s + t for s,t in zip(sources, targets)] # source + target -> "Can you translate this phrase for me? <|phrase|>, Sure thing, here is the french translation <|target|>"
    examples_tokenized, sources_tokenized = [
        _tokenize(strings, tokenizer) for strings in (complete_examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_length in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_length] = -100 # Pytorch will ignore -100 during learning in c.e.l.
    return dict(input_ids=input_ids, labels=labels)


class MyDataSet(Dataset):
    """Dataset for fine-tuning model"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, paths: str, limit=3000):
        super(MyDataSet, self).__init__()
        dataset = (
            datasets.load_dataset(
            "json",
            data_files=paths,
            split=f"train[0:{limit}]" if limit else "train",
            )
            .map(
                lambda samples: preprocess(samples, tokenizer),
                batched=True,
                batch_size=300,
            )
        )

        self.tokenizer = tokenizer
        self.input_ids = dataset["input_ids"]
        self.labels = dataset["labels"]
        # self.size = len(dataframe)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids = torch.tensor(self.input_ids[idx]),
            labels = torch.tensor(self.labels[idx])
        )

@dataclass
class MyDataCollator(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )