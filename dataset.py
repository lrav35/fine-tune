from typing import Dict, Sequence
import torch
import math
import os
import transformers
import datasets
from torch.utils.data import DataSet

def fmt_prompt(prompt):
    return f"### Instructions:\n{prompt}\n\n### Response:"

def preprocess(
        samples: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict:
    """Preprocess data for training by tokenizing"""
    sources = [f"{fmt_prompt(sources)}" for sources in samples["input"]]
    targets = [f"{translation}{tokenizer.eos_token}" for translation in samples["output"]]
    return None


class MyDataSet(DataSet):
    """Dataset for fine-tuning model"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, paths: str, limit=None):
        super(MyDataSet, self).__init__()
        dataset = (
            datasets.load_dataset(
            "json",
            data_files=paths,
            split=split=f"train[0:{limit}]" if limit else "train",
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
        