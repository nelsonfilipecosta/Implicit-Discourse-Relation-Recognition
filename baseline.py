import pandas as pd
import numpy as np
import regex as re
from sklearn import metrics
import statistics
from scipy.spatial.distance import jensenshannon

df = pd.read_csv('Data/DiscoGeM/discogem_test.csv')
df_discogem = pd.read_csv('Data/DiscoGem/discogem.csv',
                    usecols=['synchronous','precedence', 'succession', 'reason', 'result', 'arg1-as-cond', 'arg2-as-cond', 'arg1-as-goal', 'arg2-as-goal', 'arg1-as-denier', 'arg2-as-denier', 'contrast', 'similarity', 'conjunction', 'equivalence', 'arg1-as-instance', 'arg2-as-instance', 'arg1-as-detail', 'arg2-as-detail', 'arg1-as-manner', 'arg2-as-manner', 'substitution',
                    'synchronous_2', 'asynchronous', 'cause', 'condition', 'purpose', 'concession', 'contrast_2', 'similarity_2', 'conjunction_2', 'equivalence_2', 'instantiation', 'level-of-detail', 'manner', 'substitution_2',
                    'temporal', 'contingency', 'comparison', 'expansion'])


names_1 = ['temporal', 'contingency', 'comparison', 'expansion']
names_2 = ['synchronous_2', 'asynchronous', 'cause', 'condition', 'purpose', 'concession', 'contrast_2', 'similarity_2', 'conjunction_2', 'equivalence_2', 'instantiation', 'level-of-detail', 'manner', 'substitution_2']
names_3 = ['synchronous','precedence', 'succession', 'reason', 'result', 'arg1-as-cond', 'arg2-as-cond', 'arg1-as-goal', 'arg2-as-goal', 'arg1-as-denier', 'arg2-as-denier', 'contrast', 'similarity', 'conjunction', 'equivalence', 'arg1-as-instance', 'arg2-as-instance', 'arg1-as-detail', 'arg2-as-detail', 'arg1-as-manner', 'arg2-as-manner', 'substitution',]

total_sum = df_discogem['temporal'].sum() + df_discogem['contingency'].sum() +  df_discogem['comparison'].sum() + df_discogem['expansion'].sum() 

probs_1 = []
for name in names_1:
    probs_1.append(df_discogem[name].sum()/total_sum)
probs_2 = []
for name in names_2:
    probs_2.append(df_discogem[name].sum()/total_sum)
probs_3 = []
for name in names_3:
    probs_3.append(df_discogem[name].sum()/total_sum)

labels_1 = np.array(df.iloc[:,45:49])
labels_2 = np.array(df.iloc[:,29:43])
labels_3 = np.array(df.iloc[:,5:27])

probs_1 = np.array(probs_1)
probs_2 = np.array(probs_2)
probs_3 = np.array(probs_3)

def get_pred(probs, size):
    preds = np.empty(size)
    for i in range(size):
        preds[i] = np.random.uniform(0, probs[i])    
    return preds / np.sum(preds)

total_js_1 = []
total_js_2 = []
total_js_3 = []
total_f1_1 = []
total_f1_2 = []
total_f1_3 = []

for abc in range(3):

    js_1 = 0
    js_2 = 0
    js_3 = 0
    labels_l1 = []
    labels_l2 = []
    labels_l3 = []
    predictions_l1 = []
    predictions_l2 = []
    predictions_l3 = []

    for i in range(labels_1.shape[0]):
        preds_1 = get_pred(probs_1, 4)
        preds_2 = get_pred(probs_2, 14)
        preds_3 = get_pred(probs_3, 22)

        js_1 += jensenshannon(labels_1[i], preds_1, base=2)
        js_2 += jensenshannon(labels_2[i], preds_2, base=2)
        js_3 += jensenshannon(labels_3[i], preds_3, base=2)

        labels_l1.append(np.argmax(labels_1[i]).tolist())
        labels_l2.append(np.argmax(labels_2[i]).tolist())
        labels_l3.append(np.argmax(labels_3[i]).tolist())
        predictions_l1.append(np.argmax(preds_1).tolist())
        predictions_l2.append(np.argmax(preds_2).tolist())
        predictions_l3.append(np.argmax(preds_3).tolist())

    js_1 = js_1 / labels_1.shape[0]
    js_2 = js_2 / labels_1.shape[0]
    js_3 = js_3 / labels_1.shape[0]

    f1_1 = metrics.f1_score(labels_l1, predictions_l1, average='weighted', zero_division=0)
    f1_2 = metrics.f1_score(labels_l2, predictions_l2, average='weighted', zero_division=0)
    f1_3 = metrics.f1_score(labels_l3, predictions_l3, average='weighted', zero_division=0)

    total_js_1.append(js_1)
    total_js_2.append(js_2)
    total_js_3.append(js_3)
    
    total_f1_1.append(f1_1)
    total_f1_2.append(f1_2)
    total_f1_3.append(f1_3)

print(f'{statistics.mean(total_js_1):.3f}' + '   ' + f'{statistics.stdev(total_js_1):.2f}')
print(f'{statistics.mean(total_js_2):.3f}' + '   ' + f'{statistics.stdev(total_js_2):.2f}')
print(f'{statistics.mean(total_js_3):.3f}' + '   ' + f'{statistics.stdev(total_js_3):.2f}')
print(f'{100*statistics.mean(total_f1_1):.2f}' + '   ' + f'{100*statistics.stdev(total_f1_1):.2f}')
print(f'{100*statistics.mean(total_f1_2):.2f}' + '   ' + f'{100*statistics.stdev(total_f1_2):.2f}')
print(f'{100*statistics.mean(total_f1_3):.2f}' + '   ' + f'{100*statistics.stdev(total_f1_3):.2f}')