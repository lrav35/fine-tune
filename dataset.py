from typing import Dict
import torch
import math
import os
import transformers
import datasets
from torch.utils.data import DataSet


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
        