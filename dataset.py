from typing import Dict, Sequence
import torch
import math
import os
import transformers
import datasets
from torch.utils.data import Dataset

def fmt_prompt(prompt):
    return f"### Instructions:\n Can you please translate this phrase or word to french? \n {prompt}\n\n### Response:\n Yes of course! Here is a french translation of that phrase: \n"

def preprocess(
        samples: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
    """Preprocess data for training by tokenizing"""
    sources = [f"{fmt_prompt(sources)}" for sources in samples["input"]]
    targets = [f"{translation}{tokenizer.eos_token}" for translation in samples["output"]]
    complete_examples = [s + t for s,t in zip(sources, targets)]
    """tokenize examples"""
    tokenized_strings = [
        tokenizer(
            example,
            return_tensors='pt',
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) 
        for example in complete_examples
    ]
    return None


class MyDataSet(Dataset):
    """Dataset for fine-tuning model"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, paths: str, limit=None):
        super(MyDataSet, self).__init__()
        dataset = (
            datasets.load_dataset(
            "json",
            data_files=paths,
            split=f"train[0:{limit}]" if limit else "train",
            )
            .filter(
                # filter data entries
                )
            .map(
                # create a preprocessing function 
            )
        )

        self.tokenizer = tokenizer
        self.data = None 
        # self.size = len(dataframe)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        return None
        