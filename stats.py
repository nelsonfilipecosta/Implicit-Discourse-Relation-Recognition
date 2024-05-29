import pandas as pd
import regex as re

df_discogem = pd.read_csv('Data/DiscoGeM/discogem.csv')
df_discogem_test = pd.read_csv('Data/DiscoGeM/discogem_test.csv')
                         
df_pdtb = pd.read_csv('Data/PDTB-3.0/pdtb_3.csv')
df_pdtb_lin = pd.read_csv('Data/PDTB-3.0/pdtb_3_lin.csv')
df_pdtb_ji = pd.read_csv('Data/PDTB-3.0/pdtb_3_ji.csv')
df_pdtb_balanced = pd.read_csv('Data/PDTB-3.0/pdtb_3_balanced.csv')


# # !!! UNCOMMENT TO GET DATASET SIZES !!! #

# print(len(df_discogem))
# print(df_discogem['majority_level_2'].value_counts())
# print(df_discogem['majority_level_3'].value_counts())

# print(len(df_discogem_test))
# print(df_discogem_test['majority_level_2'].value_counts())
# print(df_discogem_test['majority_level_3'].value_counts())

# print(len(df_pdtb))
# print(df_pdtb['sense2'].value_counts())
# print(df_pdtb['sense3'].value_counts())

# print(len(df_pdtb_lin))
# print(df_pdtb_lin['sense2'].value_counts())
# print(df_pdtb_lin['sense3'].value_counts())

# print(len(df_pdtb_ji))
# print(df_pdtb_ji['sense2'].value_counts())
# print(df_pdtb_ji['sense3'].value_counts())

# print(len(df_pdtb_balanced))
# print(df_pdtb_balanced['sense2'].value_counts())
# print(df_pdtb_balanced['sense3'].value_counts())


# # !!! UNCOMMENT TO GET SUM OF SOFT LABEL COLUMNS !!! #

# df_discogem = pd.read_csv('Corpora/DiscoGeM/DiscoGeM_corpus/DiscoGeMcorpus_annotations_wide.csv', usecols=['majority_softlabel'])

# label_names = re.findall(r'[\w\-]+(?=\:)', df_discogem['majority_softlabel'].iloc[0])

# for i in label_names:
#     df_discogem[i] = ''

# for row in range(len(df_discogem['majority_softlabel'])):
#     label_values = re.findall(r'(?<=:)(\d\.?(\d*)?)', df_discogem['majority_softlabel'].iloc[row])
#     for i in range(len(label_names)):
#         df_discogem.loc[row, label_names[i]] = label_values[i][0]

# df_discogem.drop(columns=['majority_softlabel'], inplace=True)

# df_discogem[label_names] = df_discogem[label_names].astype(float)

# df_discogem['asynchronous'] = df_discogem['precedence'] + df_discogem['succession']
# df_discogem['cause'] = df_discogem['reason'] + df_discogem['result']
# df_discogem['condition'] = df_discogem['arg1-as-cond'] + df_discogem['arg2-as-cond']
# df_discogem['purpose'] = df_discogem['arg1-as-goal'] + df_discogem['arg2-as-goal']
# df_discogem['concession'] = df_discogem['arg1-as-denier'] + df_discogem['arg2-as-denier']
# df_discogem['instantiation'] = df_discogem['arg1-as-instance'] + df_discogem['arg2-as-instance']
# df_discogem['level-of-detail'] = df_discogem['arg1-as-detail'] + df_discogem['arg2-as-detail']
# df_discogem['manner'] = df_discogem['arg1-as-manner'] + df_discogem['arg2-as-manner']

# for label in df_discogem.columns:
#     print(label, ': ', df_discogem[label].sum())


# # !!! UNCOMMENT TO GET NUMBER OF MULTI-LABEL RELATIONS !!! #

print(len(df_pdtb))
print(df_pdtb['multi_sense'].isna().sum())
print(df_pdtb.loc[df_pdtb.multi_sense != '', 'multi_sense'].count())

print(len(df_pdtb_lin))
print(df_pdtb_lin['multi_sense'].isna().sum())
print(df_pdtb_lin.loc[df_pdtb_lin.multi_sense != '', 'multi_sense'].count())

print(len(df_pdtb_ji))
print(df_pdtb_ji['multi_sense'].isna().sum())
print(df_pdtb_ji.loc[df_pdtb_ji.multi_sense != '', 'multi_sense'].count())

print(len(df_pdtb_balanced))
print(df_pdtb_balanced['multi_sense'].isna().sum())
print(df_pdtb_balanced.loc[df_pdtb_balanced.multi_sense != '', 'multi_sense'].count())