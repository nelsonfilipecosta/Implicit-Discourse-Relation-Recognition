import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel


MODEL_NAME = 'roberta-base'
BATCH_SIZE = 16

NUMBER_OF_SENSES = {'level_1': 5,
                    'level_2': 8,
                    'level_3': 22}


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


class Multi_IDDR_Classifier(torch.nn.Module):
    'Classification model for multi-label implicit discourse relation recognition.'
    
    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        self.classifier_level_1 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_3'])
    
    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        output = last_hidden_state[:, 0]
        logits = {'classifier_level_1': self.classifier_level_1(output),
                  'classifier_level_2': self.classifier_level_2(output),
                  'classifier_level_3': self.classifier_level_3(output)}
        return logits


def create_dataloader(path):
    'Create dataloader class for multi-label implicit discourse relation recognition data splits.'
    
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


def get_loss(predictions, labels):

    loss_level_1 = loss_function(predictions['classifier_level_1'], labels['labels_level_1'])
    loss_level_2 = loss_function(predictions['classifier_level_2'], labels['labels_level_2'])
    loss_level_3 = loss_function(predictions['classifier_level_3'], labels['labels_level_3'])
    
    return loss_level_1 + loss_level_2 + loss_level_3



train_loader      = create_dataloader('Data/DiscoGeM/discogem_validation.csv')
validation_loader = create_dataloader('Data/DiscoGeM/discogem_validation.csv')
test_loader       = create_dataloader('Data/DiscoGeM/discogem_validation.csv')




model = Multi_IDDR_Classifier(MODEL_NAME, NUMBER_OF_SENSES)
loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
# loss_function = torch.nn.L1Loss(reduction='mean') # try this loss function

for batch_idx, batch in enumerate(train_loader):

    # forward pass
    model_output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    loss = get_loss(model_output, batch)