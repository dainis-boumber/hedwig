import pandas as pd
import numpy as np
from pathlib import Path
from pandas.api.types import is_string_dtype
from typing import List, Set, Dict, Tuple, Optional, Union

class NewDatasetProcessor:

    def __init__(self):
        pass
    
    def preprocess(text):
        # remove URL's from train and test
        text=ftfy.fix_text(text)
        text=textacy.preprocess.preprocess_text(text,
                                                no_urls=True,
                                                no_emails=True,
                                                no_phone_numbers=True,
                                                no_accents=True,
                                                )
        return ' '.join(text.split()).strip()

    def prepare_csv(csv:str, sep:str=',', header='infer', text_cols:List=[, label_cols:List, low_memory=False):
        ''' Converts csv to tsv that is readable by Hedwig
            Params:
                csv - relative or full filname, for example myfile.csv or ../data/files/myfile.csv or /datasets/xyz/myfile.csv
                sep - what is used to separate fields, for example '\t' or ','. Default is ','
                header - does the csv have a header file? default is 'infer', can be False
                text_cols - which column(s) contain text
                label_cols - which column(s) has the labels
            TODO: optimize for speed and memory use
        '''
    
        if path is None:
            path = Path('.')
        else:
            path = Path(path)
        df = pd.read_csv(path / csv_file_name, sep=sep, header=header)
        df[text_cols] = df[text_cols].str.replace("\n", " ")

        if len(label_cols) > 1:
            df['label'] = df.apply(lambda row: ''.join([str(val) for val in row[label_cols]]), axis=1)

        df.to_csv(path/'tmp.tsv', sep='\t', header=False, index=False)
