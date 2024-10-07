import os
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


SPLIT = sys.argv[1]
if SPLIT not in ['discogem', 'lin', 'ji', 'balanced', 'cross']:
    print('Type a valid test split: discogem, lin, ji, balanced or cross.')
    exit()


if SPLIT == 'discogem':
    folder = 'DiscoGeM'
elif SPLIT == 'lin':
    folder = 'PDTB_Lin'
elif SPLIT == 'ji':
    folder = 'PDTB_Ji'
elif SPLIT == 'balanced':
    folder = 'PDTB_Balanced'
else:
    folder = 'PDTB_Cross'


MODEL = sys.argv[2]
if MODEL not in ['1', '2', '3']:
    print('Type a valid test iteration: 1, 2, or 3.')
    exit()
MODEL = int(MODEL)


def listdir_nohidden(path):
    folders = os.listdir(path)
    for f in folders:
        if f.startswith('.'):
            folders.remove(f)
    return folders


def plot_cm(level, split, labels, predictions):

    if level == 1:
        senses = ['Temporal', 'Contigency', 'Comparison', 'Expansion']
    elif level == 2 and split == 'discogem':
        senses = ['Synchronous', 'Asynchronous', 'Cause', 'Concession', 'Contrast', 'Similarity', 'Conjunction', 'Instantiation', 'Level-of-Detail', 'Substitution']
    elif level == 2 and split != 'discogem':
        senses = ['Synchronous', 'Asynchronous', 'Cause', 'Condition', 'Purpose', 'Concession', 'Contrast', 'Similarity', 'Conjunction', 'Equivalence', 'Instantiation', 'Level-of-Detail', 'Manner', 'Substitution']
    
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

    if split == 'discogem':
        plt.title('DiscoGeM (Test): Level-' + str(level))
    elif split == 'lin':
        plt.title('PDTB (Lin): Level-' + str(level))
    elif split == 'ji':
        plt.title('PDTB (Ji): Level-' + str(level))
    elif split == 'balanced':
        plt.title('PDTB (Balanced): Level-' + str(level))
    elif split == 'cross':
        plt.title('PDTB (Cross): Level-' + str(level))

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    plt.xticks(ticks=np.arange(len(senses)) + 0.5, labels=senses, rotation=90)
    plt.yticks(ticks=np.arange(len(senses)) + 0.5, labels=senses, rotation=0)

    plt.tight_layout()
    plt.savefig('Results/Confusion Matrices/confusion_matrix_'+ split + '_l' + str(level) + '_'+ str(MODEL) +'.png', format='png', dpi=900, bbox_inches='tight')
    plt.show()


if not os.path.exists('Results/Confusion Matrices'):
    os.makedirs('Results/Confusion Matrices')

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

labels_1 = []
predictions_1 = []
f1_scores_1 = []

for i in range(len(listdir_nohidden('Results/'+folder))):
    labels_1.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/labels_l1.txt').astype(int))
    predictions_1.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/predictions_l1.txt').astype(int))

plot_cm(1, SPLIT, labels_1[MODEL-1], predictions_1[MODEL-1])

labels_2 = []
predictions_2 = []
f1_scores_2 = []

for i in range(len(listdir_nohidden('Results/'+folder))):
    labels_2.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/labels_l2.txt').astype(int))
    predictions_2.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/predictions_l2.txt').astype(int))

plot_cm(2, SPLIT, labels_2[MODEL-1], predictions_2[MODEL-1])