import os
import sys
import numpy as np
import statistics
from sklearn.metrics import f1_score


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


def listdir_nohidden(path):
    folders = os.listdir(path)
    for f in folders:
        if f.startswith('.'):
            folders.remove(f)
    return folders


def get_counts_l1(predictions_1):
    
    temporal    = [0]*len(predictions_1)
    contingency = [0]*len(predictions_1)
    comparison  = [0]*len(predictions_1)
    expansion   = [0]*len(predictions_1)

    for i in range(len(predictions_1)):
        _, counts = np.unique(predictions_1[i], return_counts=True)

        temporal[i]    = counts[0]
        contingency[i] = counts[1]
        comparison[i]  = counts[2]
        expansion[i]   = counts[3]

    return temporal, contingency, comparison, expansion


def get_counts_l2(predictions_2):
    
    synchronous   = [0]*len(predictions_1)
    asynchronous  = [0]*len(predictions_1)
    cause         = [0]*len(predictions_1)
    condition     = [0]*len(predictions_1)
    purpose       = [0]*len(predictions_1)
    concession    = [0]*len(predictions_1)
    contrast      = [0]*len(predictions_1)
    similarity    = [0]*len(predictions_1)
    conjunction   = [0]*len(predictions_1)
    equivalence   = [0]*len(predictions_1)
    instantiation = [0]*len(predictions_1)
    detail        = [0]*len(predictions_1)
    manner        = [0]*len(predictions_1)
    substitution  = [0]*len(predictions_1)

    for i in range(len(predictions_2)):
        unique, counts = np.unique(predictions_2[i], return_counts=True)
        
        for j in range(unique.shape[0]):
            #synchronous
            if unique[j] == 0:
                synchronous[i]   = counts[j]
            # asynchronous
            elif unique[j] == 1:
                asynchronous[i]  = counts[j]
            # cause
            elif unique[j] == 2:
                cause[i]         = counts[j]
            # condition
            elif unique[j] == 3:
                condition[i]     = counts[j]
            # purpose
            elif unique[j] == 4:
                purpose[i]       = counts[j]
            # concession
            elif unique[j] == 5:
                concession[i]    = counts[j]
            # contrast
            elif unique[j] == 6:
                contrast[i]      = counts[j]
            # similarity
            elif unique[j] == 7:
                similarity[i]    = counts[j]
            # conjunction
            elif unique[j] == 8:
                conjunction[i]   = counts[j]
            # equivalence
            elif unique[j] == 9:
                equivalence[i]   = counts[j]
            # instantiation
            elif unique[j] == 10:
                instantiation[i] = counts[j]
            # detail
            elif unique[j] == 11:
                detail[i]        = counts[j]
            # manner
            elif unique[j] == 12:
                manner[i]        = counts[j]
            # substitution
            else:
                substitution[i]  = counts[j]

    return synchronous, asynchronous, cause, condition, purpose, concession, contrast, similarity, conjunction, equivalence, instantiation, detail, manner, substitution


def get_inconsistencies_l1(predictions_1, predictions_2):
    
    temporal    = [0]*len(predictions_1)
    contingency = [0]*len(predictions_1)
    comparison  = [0]*len(predictions_1)
    expansion   = [0]*len(predictions_1)

    for i in range(len(predictions_1)):
        predictions_1_np = np.array(predictions_1[i])
        predictions_2_np = np.array(predictions_2[i])

        for j in range(predictions_1_np.shape[0]):
            # temporal
            if predictions_1_np[j] == 0:
                if predictions_2_np[j] not in [0,1]:
                    temporal[i] += 1
            # contingency
            if predictions_1_np[j] == 1:
                if predictions_2_np[j] not in [2,3,4]:
                    contingency[i] += 1
            # comparison
            if predictions_1_np[j] == 2:
                if predictions_2_np[j] not in [5,6,7]:
                    comparison[i] += 1
            # expansion
            if predictions_1_np[j] == 3:
                if predictions_2_np[j] not in [8,9,10,11,12,13]:
                    expansion[i] += 1

    return temporal, contingency, comparison, expansion


def get_inconsistencies_l2(predictions_1, predictions_2):
    
    synchronous   = [0]*len(predictions_1)
    asynchronous  = [0]*len(predictions_1)
    cause         = [0]*len(predictions_1)
    condition     = [0]*len(predictions_1)
    purpose       = [0]*len(predictions_1)
    concession    = [0]*len(predictions_1)
    contrast      = [0]*len(predictions_1)
    similarity    = [0]*len(predictions_1)
    conjunction   = [0]*len(predictions_1)
    equivalence   = [0]*len(predictions_1)
    instantiation = [0]*len(predictions_1)
    detail        = [0]*len(predictions_1)
    manner        = [0]*len(predictions_1)
    substitution  = [0]*len(predictions_1)

    for i in range(len(predictions_1)):
        predictions_1_np = np.array(predictions_1[i])
        predictions_2_np = np.array(predictions_2[i])

        for j in range(predictions_1_np.shape[0]):
            # synchronous
            if predictions_2_np[j] == 0:
                if predictions_1_np[j] != 0:
                    synchronous[i] += 1
            # asynchronous
            if predictions_2_np[j] == 1:
                if predictions_1_np[j] != 0:
                    asynchronous[i] += 1
            # cause
            if predictions_2_np[j] == 2:
                if predictions_1_np[j] != 1:
                    cause[i] += 1
            # condition
            if predictions_2_np[j] == 3:
                if predictions_1_np[j] != 1:
                    condition[i] += 1
            # purpose
            if predictions_2_np[j] == 4:
                if predictions_1_np[j] != 1:
                    purpose[i] += 1
            # concession
            if predictions_2_np[j] == 5:
                if predictions_1_np[j] != 2:
                    concession[i] += 1
            # contrast
            if predictions_2_np[j] == 6:
                if predictions_1_np[j] != 2:
                    contrast[i] += 1
            # similarity
            if predictions_2_np[j] == 7:
                if predictions_1_np[j] != 2:
                    similarity[i] += 1
            # conjunction
            if predictions_2_np[j] == 8:
                if predictions_1_np[j] != 3:
                    conjunction[i] += 1
            # equivalence
            if predictions_2_np[j] == 9:
                if predictions_1_np[j] != 3:
                    equivalence[i] += 1
            # instantiation
            if predictions_2_np[j] == 10:
                if predictions_1_np[j] != 3:
                    instantiation[i] += 1
            # detail
            if predictions_2_np[j] == 11:
                if predictions_1_np[j] != 3:
                    detail[i] += 1
            # manner
            if predictions_2_np[j] == 12:
                if predictions_1_np[j] != 3:
                    manner[i] += 1
            # substitution
            if predictions_2_np[j] == 13:
                if predictions_1_np[j] != 3:
                    substitution[i] += 1

    return synchronous, asynchronous, cause, condition, purpose, concession, contrast, similarity, conjunction, equivalence, instantiation, detail, manner, substitution


def get_percentages(sense, sense_count):

    percentage = [0]*len(sense)

    for i in range(len(sense)):
        if sense_count[i] != 0:
            percentage[i] = round(sense[i]/sense_count[i]*100, 2)
    
    return percentage


labels_1 = []
predictions_1 = []
f1_scores_1 = []

for i in range(len(listdir_nohidden('Results/'+folder))):
    labels_1.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/labels_l1.txt').astype(int))
    predictions_1.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/predictions_l1.txt').astype(int))

    f1_scores_1.append(f1_score(np.array(labels_1[i]), np.array(predictions_1[i]), average=None).tolist())

f1_scores_1 = np.array(f1_scores_1)

for i in range(f1_scores_1.shape[1]):
    print(str(round(np.mean(f1_scores_1[:,i])*100, 2)) + ' +- ' + str(round(np.std(f1_scores_1[:,i])*100, 2)))

print('\n')

################

labels_2 = []
predictions_2 = []
f1_scores_2 = []

for i in range(len(listdir_nohidden('Results/'+folder))):
    labels_2.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/labels_l2.txt').astype(int))
    predictions_2.append(np.loadtxt('Results/'+folder+'/'+folder+'_'+str(i)+'/predictions_l2.txt').astype(int))

    f1_scores_2.append(f1_score(np.array(labels_2[i]), np.array(predictions_2[i]), average=None).tolist())

f1_scores_2 = np.array(f1_scores_2)

for i in range(f1_scores_2.shape[1]):
    print(str(round(np.mean(f1_scores_2[:,i])*100, 2)) + ' +- ' + str(round(np.std(f1_scores_2[:,i])*100, 2)))

print('\n')

################

temporal, contingency, comparison, expansion = get_inconsistencies_l1(predictions_1, predictions_2)
temporal_count, contingency_count, comparison_count, expansion_count = get_counts_l1(predictions_1)

print(f'Temporal: {statistics.mean(temporal):.2f} +/- {statistics.stdev(temporal):.2f}')
print(f'Contingency: {statistics.mean(contingency):.2f} +/- {statistics.stdev(contingency):.2f}')
print(f'Comparison: {statistics.mean(comparison):.2f} +/- {statistics.stdev(comparison):.2f}')
print(f'Expansion: {statistics.mean(expansion):.2f} +/- {statistics.stdev(expansion):.2f}')

print('\n')

synchronous, asynchronous, cause, condition, purpose, concession, contrast, similarity, conjunction, equivalence, instantiation, detail, manner, substitution = get_inconsistencies_l2(predictions_1, predictions_2)
synchronous_count, asynchronous_count, cause_count, condition_count, purpose_count, concession_count, contrast_count, similarity_count, conjunction_count, equivalence_count, instantiation_count, detail_count, manner_count, substitution_count = get_counts_l2(predictions_2)

print(f'Synchronous: {statistics.mean(synchronous):.2f} +/- {statistics.stdev(synchronous):.2f}')
print(f'Asynchronous: {statistics.mean(asynchronous):.2f} +/- {statistics.stdev(asynchronous):.2f}')
print(f'Cause: {statistics.mean(cause):.2f} +/- {statistics.stdev(cause):.2f}')
print(f'Condition: {statistics.mean(condition):.2f} +/- {statistics.stdev(condition):.2f}')
print(f'Purpose: {statistics.mean(purpose):.2f} +/- {statistics.stdev(purpose):.2f}')
print(f'Concession: {statistics.mean(concession):.2f} +/- {statistics.stdev(concession):.2f}')
print(f'Contrast: {statistics.mean(contrast):.2f} +/- {statistics.stdev(contrast):.2f}')
print(f'Similarity: {statistics.mean(similarity):.2f} +/- {statistics.stdev(similarity):.2f}')
print(f'Conjunction: {statistics.mean(conjunction):.2f} +/- {statistics.stdev(conjunction):.2f}')
print(f'Equivalence: {statistics.mean(equivalence):.2f} +/- {statistics.stdev(equivalence):.2f}')
print(f'Instantiation: {statistics.mean(instantiation):.2f} +/- {statistics.stdev(instantiation):.2f}')
print(f'Detail: {statistics.mean(detail):.2f} +/- {statistics.stdev(detail):.2f}')
print(f'Manner: {statistics.mean(manner):.2f} +/- {statistics.stdev(manner):.2f}')
print(f'Substitution: {statistics.mean(substitution):.2f} +/- {statistics.stdev(substitution):.2f}')

print('\n')

################

temporal_ave    = get_percentages(temporal, temporal_count)
contingency_ave = get_percentages(contingency, contingency_count)
comparison_ave  = get_percentages(comparison, comparison_count)
expansion_ave   = get_percentages(expansion, expansion_count)

print(f'Temporal: {statistics.mean(temporal_ave):.2f} +/- {statistics.stdev(temporal_ave):.2f}')
print(f'Contingency: {statistics.mean(contingency_ave):.2f} +/- {statistics.stdev(contingency_ave):.2f}')
print(f'Comparison: {statistics.mean(comparison_ave):.2f} +/- {statistics.stdev(comparison_ave):.2f}')
print(f'Expansion: {statistics.mean(expansion_ave):.2f} +/- {statistics.stdev(expansion_ave):.2f}')

print('\n')

synchronous_ave   = get_percentages(synchronous, synchronous_count)
asynchronous_ave  = get_percentages(asynchronous, asynchronous_count)
cause_ave         = get_percentages(cause, cause_count)
condition_ave     = get_percentages(condition, condition_count)
purpose_ave       = get_percentages(purpose, purpose_count)
concession_ave    = get_percentages(concession, concession_count)
contrast_ave      = get_percentages(contrast, contrast_count)
similarity_ave    = get_percentages(similarity, similarity_count)
conjunction_ave   = get_percentages(conjunction, conjunction_count)
equivalence_ave   = get_percentages(equivalence, equivalence_count)
instantiation_ave = get_percentages(instantiation, instantiation_count)
detail_ave        = get_percentages(detail, detail_count)
manner_ave        = get_percentages(manner, manner_count)
substitution_ave  = get_percentages(substitution, substitution_count)

print(f'Synchronous: {statistics.mean(synchronous_ave):.2f} +/- {statistics.stdev(synchronous_ave):.2f}')
print(f'Asynchronous: {statistics.mean(asynchronous_ave):.2f} +/- {statistics.stdev(asynchronous_ave):.2f}')
print(f'Cause: {statistics.mean(cause_ave):.2f} +/- {statistics.stdev(cause_ave):.2f}')
print(f'Condition: {statistics.mean(condition_ave):.2f} +/- {statistics.stdev(condition_ave):.2f}')
print(f'Purpose: {statistics.mean(purpose_ave):.2f} +/- {statistics.stdev(purpose_ave):.2f}')
print(f'Concession: {statistics.mean(concession_ave):.2f} +/- {statistics.stdev(concession_ave):.2f}')
print(f'Contrast: {statistics.mean(contrast_ave):.2f} +/- {statistics.stdev(contrast_ave):.2f}')
print(f'Similarity: {statistics.mean(similarity_ave):.2f} +/- {statistics.stdev(similarity_ave):.2f}')
print(f'Conjunction: {statistics.mean(conjunction_ave):.2f} +/- {statistics.stdev(conjunction_ave):.2f}')
print(f'Equivalence: {statistics.mean(equivalence_ave):.2f} +/- {statistics.stdev(equivalence_ave):.2f}')
print(f'Instantiation: {statistics.mean(instantiation_ave):.2f} +/- {statistics.stdev(instantiation_ave):.2f}')
print(f'Detail: {statistics.mean(detail_ave):.2f} +/- {statistics.stdev(detail_ave):.2f}')
print(f'Manner: {statistics.mean(manner_ave):.2f} +/- {statistics.stdev(manner_ave):.2f}')
print(f'Substitution: {statistics.mean(substitution_ave):.2f} +/- {statistics.stdev(substitution_ave):.2f}')