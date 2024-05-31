import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel


MODEL_NAME = 'roberta-base'
BATCH_SIZE = 16


class Multi_IDDR_Dataset(torch.utils.data.Dataset):
    'Dataset class for multi-label implicit discourse relation regognition classification tasks.'

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels_level_3'] = torch.tensor(self.labels[idx,0:22])  # level-3 columns
        item['labels_level_2'] = torch.tensor(self.labels[idx,22:30]) # level-2 columns
        item['labels_level_1'] = torch.tensor(self.labels[idx,30:35]) # level-1 columns
        return item

    def __len__(self):
        return self.labels.shape[0]


def create_dataloader(path):
    'Create dataloader class for multi-label implicit discourse relation regognition data splits.'
    
    # read pre-processed data
    df = pd.read_csv(path)

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformers.logging.set_verbosity_error() # remove model checkpoint warning

    # prepare text encodings and labels
    encodings = tokenizer(list(df['arg1_arg2']), truncation=True, padding=True)
    labels = np.hstack((np.array(df.iloc[:,5:27]),   # level-3 columns
                        np.array(df.iloc[:,28:36]),  # level-2 columns
                        np.array(df.iloc[:,37:42]))) # level-1 columns

    # generate datasets
    dataset = Multi_IDDR_Dataset(encodings, labels)

    # generate dataloaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader





train_loader      = create_dataloader('Data/DiscoGeM/discogem_validation.csv')
validation_loader = create_dataloader('Data/DiscoGeM/discogem_validation.csv')
test_loader       = create_dataloader('Data/DiscoGeM/discogem_validation.csv')


# print(train_loader.__getitem__(0))