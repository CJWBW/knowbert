import pandas as pd
from collections import Counter
from pathlib import Path


class DataProcessor:
    TRAIN_PATH = Path(__file__).parent / "../data/train.tsv"
    TEST_PATH = Path(__file__).parent / "../data/test.tsv"
    VAL_PATH = Path(__file__).parent / "../data/valid.tsv"

    @staticmethod
    def load_dataset():
        train_df0 = pd.read_csv(DataProcessor.TRAIN_PATH, sep="\t", header=None)
        train_df = train_df0[:500]  # for experimenting with small dataset
        test_df0 = pd.read_csv(DataProcessor.TEST_PATH, sep="\t", header=None)
        test_df = test_df0[:50]
        val_df0 = pd.read_csv(DataProcessor.VAL_PATH, sep="\t", header=None)
        val_df = val_df0[:50]

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()
        val_df = val_df.to_numpy()

        labels = {'train': [train_df[i][1] for i in range(len(train_df))],
                  'test': [test_df[i][1] for i in range(len(test_df))],
                  'validation': [val_df[i][1] for i in range(len(val_df))]}
        statements = {'train': [train_df[i][2] for i in range(len(train_df))],
                      'test': [test_df[i][2] for i in range(len(test_df))],
                      'validation': [val_df[i][2] for i in range(len(val_df))]}

        print('Label distribution in training dataset')
        print(Counter(labels['train']))
        print(Counter(labels['test']))
        print(Counter(labels['validation']))

        return labels, statements

    @staticmethod
    def convert_labels(num_labels, labels):
        if not num_labels == 6 and not num_labels == 2:
            print('Invalid number of labels. The number of labels should be either 2 or 6')
        # only consider binary case now
        encoded_labels = [0] * len(labels)
        if num_labels == 2:
            for i in range(len(labels)):
                if labels[i] in ['true', 'mostly-true', 'half-true']:
                    encoded_labels[i] = 0
                elif labels[i] in ['barely-true', 'false', 'pants-fire']:
                    encoded_labels[i] = 1
                else:
                    print('Incorrect label')
        else:
            for i in range(len(labels)):
                if labels[i] == 'true':
                    encoded_labels[i] = 0
                elif labels[i] == 'mostly-true':
                    encoded_labels[i] = 1
                elif labels[i] == 'half-true':
                    encoded_labels[i] = 2
                elif labels[i] == 'barely-true':
                    encoded_labels[i] = 3
                elif labels[i] == 'false':
                    encoded_labels[i] = 4
                elif labels[i] == 'pants-fire':
                    encoded_labels[i] = 5
                else:
                    print('Incorrect label')
        return encoded_labels
