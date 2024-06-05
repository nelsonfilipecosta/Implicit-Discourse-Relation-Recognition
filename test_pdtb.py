import wandb
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn import metrics


MODEL_NAME = 'roberta-base'
BATCH_SIZE = 16

NUMBER_OF_SENSES = {'level_1': 4,
                    'level_2': 14,
                    'level_3': 22}


WANDB = 0 # set 1 for logging and 0 for local runs
if WANDB == 1:
    wandb.login()
    run = wandb.init(project = 'IDRR', config = {'Model': MODEL_NAME})


class Multi_IDDR_Dataset(torch.utils.data.Dataset):
    'Dataset class for multi-label implicit discourse relation regognition classification tasks.'

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels_level_2'] = torch.tensor(self.labels[idx,0:14])  # level-2 columns
        item['labels_level_1'] = torch.tensor(self.labels[idx,14:18]) # level-1 columns
        return item

    def __len__(self):
        return self.labels.shape[0]


class Multi_IDDR_Classifier(torch.nn.Module):
    'Multi-head classification model for multi-label implicit discourse relation recognition.'
    
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


def log_wandb(mode, f1_score_1, precision_1, recall_1, f1_score_2, precision_2, recall_2, f1_score_3, precision_3, recall_3):
    'Log metrics on Weights & Biases.'

    wandb.log({mode + ' F1 Score (Level-1)'   : f1_score_1,
               mode + ' Precision (Level-1)'  : precision_1,
               mode + ' Recall (Level-1)'     : recall_1,
               mode + ' F1 Score (Level-2)'   : f1_score_2,
               mode + ' Precision (Level-2)'  : precision_2,
               mode + ' Recall (Level-2)'     : recall_2})


def create_dataloader(path):
    'Create dataloader class for multi-label implicit discourse relation recognition data splits.'
    
    # read pre-processed data
    df = pd.read_csv(path)

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformers.logging.set_verbosity_error() # remove model checkpoint warning
    
    # prepare text encodings and labels
    encodings = tokenizer(list(df['arg1_arg2']), truncation=True, padding=True)
    labels = np.hstack((np.array(df.iloc[:,18:32]),  # level-2 columns
                        np.array(df.iloc[:,33:37]))) # level-1 columns
    
    # generate datasets
    dataset = Multi_IDDR_Dataset(encodings, labels)

    # generate dataloaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


def get_single_metrics(level, labels, predictions):
    'Get f1-score, precision and recall metrics for single-label classification.'

    f1_score    = metrics.f1_score(labels, predictions, average='weighted', zero_division=0)
    precision   = metrics.precision_score(labels, predictions, average='weighted', zero_division=0)
    recall      = metrics.recall_score(labels, predictions, average='weighted', zero_division=0)
    
    print(level + f' || F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
    
    return f1_score, precision, recall


def test_loop(mode, dataloader):
    'Validation and test loop of the classification model.'

    # group metric across all batches
    labels_l1 = []
    labels_l2 = []
    predictions_l1 = []
    predictions_l2 = []

    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            # forward pass
            model_output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            labels_l1.extend(torch.argmax(batch['labels_level_1'], dim=1).tolist())
            labels_l2.extend(torch.argmax(batch['labels_level_2'], dim=1).tolist())
            predictions_l1.extend(torch.argmax(model_output['classifier_level_1'], dim=1).tolist())
            predictions_l2.extend(torch.argmax(model_output['classifier_level_2'], dim=1).tolist())

            print(f'{batch_idx:04d}/{len(dataloader):04d}')

    f1_score_1, precision_1, recall_1 = get_single_metrics('Level-1', np.array(labels_l1), np.array(predictions_l1))
    f1_score_2, precision_2, recall_2 = get_single_metrics('Level-2', np.array(labels_l2), np.array(predictions_l2))
    
    if WANDB == 1:
        log_wandb(mode, f1_score_1, precision_1, recall_1, f1_score_2, precision_2, recall_2)


lin_loader      = create_dataloader('Data/PDTB-3.0/pdtb_3_lin.csv')
ji_loader       = create_dataloader('Data/PDTB-3.0/pdtb_3_ji.csv')
balanced_loader = create_dataloader('Data/PDTB-3.0/pdtb_3_balanced.csv')

model = Multi_IDDR_Classifier(MODEL_NAME, NUMBER_OF_SENSES)
model.load_state_dict(torch.load('Model/'+MODEL_NAME+'.pth'))

print('Testing Lin split...')
test_loop('Lin', lin_loader)

print('Testing Ji split...')
test_loop('Ji', ji_loader)

print('Testing Balanced split...')
test_loop('Balanced', balanced_loader)