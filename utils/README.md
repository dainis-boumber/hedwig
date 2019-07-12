You can use both `add_dataset.ipynb` and `add_dataset.py` to pre-process data and add new datasets from existing pandas-saved csv files,
or directly from pandas dataframes, as well as most other formats.

Command line args for the `add_dataset.py` script:

```
parser.add_argument("-d", "--dataset-name", dest="dataset_name",
                        help="name of the dataset being created", metavar="DATASET_NAME")
parser.add_argument("-n", "--num-labels", dest="num_labels_in_col",
                        help="number of labels contained in the column in case there is one label column (could be either 1 or labels could be a string stored in a column)",
                        metavar="NUM_LABELS_IN_COL")
parser.add_argument("-x", "--train", dest="train",
                        help="name of the data file to be converted to train.tsv", metavar="TRAIN")
parser.add_argument("-v", "--dev", dest="dev",
                        help="name of the data file to be converted to dev.tsv", metavar="DEV")
parser.add_argument("-t", "--test", dest="test",
                        help="name of the data file to be converted to test.tsv", metavar="TEST")
```
