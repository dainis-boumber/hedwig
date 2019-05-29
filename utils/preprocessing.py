import pandas as pd
import numpy as np
from pathlib import Path
from pandas.api.types import is_string_dtype
from typing import List, Set, Dict, Tuple, Optional, Union

class NewDatasetProcessor:

    def __init__(self):
        pass

    def prepare_dataset(dataset):
    """TODO: Make this into a standard first step of adding a new dataset"""
        pass
        
    def prepare_csv(path:Optional[str], csv_file:str, sep:str, header='infer', text_cols:List[str], label_cols:List[str], low_memory=False):
    """TODO: optimize for speed and memory use
        Process a csv (usually a Pandas-generated one) in such a way that torchtext can handle it
    """
        if path is None:
            path = Path('.')
        else:
            path = Path(path)
        df = pd.read_csv(path / csv_file_name, sep=sep, header=header)
        df[text_cols] = df[text_cols].str.replace("\n", " ")

        if len(label_cols) > 1:
            df['label'] = df.apply(lambda row: ''.join([str(val) for val in row[label_cols]]), axis=1)

        df.to_csv(path/'tmp.tsv', sep='\t', header=False, index=False)
