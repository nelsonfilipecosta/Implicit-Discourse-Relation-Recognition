import os
import time
import pandas as pd
import regex as re
from sklearn.model_selection import ShuffleSplit

if not os.path.exists('Data'):
   os.makedirs('Data')
if not os.path.exists('Data/PDTB-3.0'):
   os.makedirs('Data/PDTB-3.0')

start_time = time.time()
print(f"Preparing PDTB 3.0 corpus...")

gold_list = []
gold_path = 'Corpora/PDTB-3.0/gold'
for folder in os.listdir(gold_path):
   if not folder.startswith('.'):
        for file in os.listdir(gold_path+'/'+folder):
            for line in open(gold_path+'/'+folder+'/'+file, 'r').readlines():
                if line.startswith('Implicit'):
                    gold_list.append(line.rstrip()+'||folder_'+folder+'||file_'+file)

connective_list = []
multi_connective_list = []
sense_list = []
multi_sense_list = []
arg1_span_list = []
arg2_span_list = []
folder_list = []
file_list = []
for discourse_relation in gold_list:
    connective = re.search(r'(?<=(([\w\s\-\+\.\;\(\)\/\']+)?\|){7})([\w\s\']+)(?=\|)', discourse_relation)
    connective_list.append(connective.group(0))
    sense = re.search(r'(?<=(([\w\s\-\+\.\;\(\)\/\']+)?\|){8})([\w\s\.\+\-]+)(?=\|)', discourse_relation)
    sense_list.append(sense.group(0))
    multi_connective = re.search(r'(?<=(I(([\w\s\-\+\.\;\(\)\/\']+)?\|){10}))([a-zA-Z\s\']+)(?=\|)', discourse_relation)
    if multi_connective == None:
        multi_connective_list.append('')
        multi_sense_list.append('')
    else:
        multi_connective_list.append(multi_connective.group(0))
        multi_sense = re.search(r'(?<=(I(([\w\s\-\+\.\;\(\)\/\']+)?\|){11}))([a-zA-Z12\s\+\-]+\.[a-zA-Z12\s\+\-]+\.?[a-zA-Z12\s\+\-]+?)(?=\|)', discourse_relation)
        multi_sense_list.append(multi_sense.group(0))
    arg1_span = re.search(r'(?<=(([\w\s\-\+\.\;\(\)\/\']+)?\|){14})([\d\.\;]+)(?=\|)', discourse_relation)
    arg1_span_list.append(arg1_span.group(0))
    arg2_span = re.search(r'(?<=(([\w\s\-\+\.\;\(\)\/\']+)?\|){20})([\d\.\;]+)(?=\|)', discourse_relation)
    arg2_span_list.append(arg2_span.group(0))
    folder = re.search(r'(?<=\|\|folder_)\d+', discourse_relation)
    folder_list.append(folder.group(0))
    file = re.search(r'(?<=\|\|file_)wsj_\d+', discourse_relation)
    file_list.append(file.group(0))

sense1_list = []
sense2_list = []
sense3_list = []
sense_final = []
for sense in sense_list:
    if sense == '':
        sense1_list.append('')
        sense2_list.append('')
        sense3_list.append('')
        sense_final.append('')
    else:
        sense_1 = re.search(r'\w*', sense)
        sense1_list.append(sense_1.group(0))
        sense_2 = re.search(r'(?<=\.)[\w\-\+]*', sense)
        sense2_list.append(sense_2.group(0))
        sense_3 = re.search(r'(?<=(\.[\w\-\+]*\.)).*', sense)
        if sense_3 == None:
            sense3_list.append('')
            sense_final.append(sense_2.group(0))
        else:
            sense3_list.append(sense_3.group(0))
            sense_final.append(sense_3.group(0))

multi_sense1_list = []
multi_sense2_list = []
multi_sense3_list = []
multi_sense_final = []
for multi_sense in multi_sense_list:
    if multi_sense == '':
        multi_sense1_list.append('')
        multi_sense2_list.append('')
        multi_sense3_list.append('')
        multi_sense_final.append('')
    else:
        multi_sense1 = re.search(r'\w*', multi_sense)
        multi_sense1_list.append(multi_sense1.group(0))
        multi_sense2 = re.search(r'(?<=\.)[\w\-\+]*', multi_sense)
        multi_sense2_list.append(multi_sense2.group(0))
        multi_sense3 = re.search(r'(?<=(\.[\w\-\+]*\.)).*', multi_sense)
        if multi_sense3 == None:
            multi_sense3_list.append('')
            multi_sense_final.append(multi_sense2.group(0))
        else:
            multi_sense3_list.append(multi_sense3.group(0))
            multi_sense_final.append(multi_sense3.group(0))

raw_path = 'Corpora/PDTB-3.0/raw'
arg1_list = ['' for i in range(len(gold_list))]
arg2_list = ['' for i in range(len(gold_list))]
for i in range(len(gold_list)):
    arg1_begin = re.findall(r'\d+(?=\.\.)', arg1_span_list[i])
    arg1_end = re.findall(r'(?<=\.\.)\d+', arg1_span_list[i])
    arg2_begin = re.findall(r'\d+(?=\.\.)', arg2_span_list[i])
    arg2_end = re.findall(r'(?<=\.\.)\d+', arg2_span_list[i])
    for j in range(len(arg1_begin)):
        if j < len(arg1_begin)-1:
            argument_sep = ' '
        else:
            argument_sep = ''
        with open(raw_path+'/'+folder_list[i]+'/'+file_list[i], 'r', encoding = 'ISO-8859-1') as text:
            arg1_list[i] += text.read().rstrip()[int(arg1_begin[j]):int(arg1_end[j])] + argument_sep
    for k in range(len(arg2_begin)):
        if k < len(arg2_begin)-1:
            argument_sep = ' '
        else:
            argument_sep = ''
        with open(raw_path+'/'+folder_list[i]+'/'+file_list[i], 'r', encoding = 'ISO-8859-1') as text:
            arg2_list[i] += text.read().rstrip()[int(arg2_begin[k]):int(arg2_end[k])] + argument_sep

df = pd.DataFrame({'folder':            folder_list,
                   'file':              file_list,
                   'arg1':              arg1_list,
                   'arg2':              arg2_list,
                   'connective':        connective_list,
                   'sense':             sense_list,
                   'sense_final':       sense_final,
                   'sense1':            sense1_list,
                   'sense2':            sense2_list,
                   'sense3':            sense3_list,
                   'multi_connective':  multi_connective_list,
                   'multi_sense':       multi_sense_list,
                   'multi_sense_final': multi_sense_final,
                   'multi_sense1':      multi_sense1_list,
                   'multi_sense2':      multi_sense2_list,
                   'multi_sense3':      multi_sense3_list})

df['arg1_arg2'] = df['arg1'].copy() + ' ' + df['arg2'].copy()

df = df[['folder', 'file', 'arg1', 'arg2', 'arg1_arg2', 'connective', 'sense', 'sense_final', 'sense1', 'sense2', 'sense3',
         'multi_connective', 'multi_sense', 'multi_sense_final', 'multi_sense1', 'multi_sense2', 'multi_sense3']]

df = df[df['sense2'].isin(['Synchronous', 'Asynchronous', 'Cause', 'Cause+Belief', 'Condition', 'Purpose', 'Concession', 'Contrast',
                           'Conjunction', 'Equivalence', 'Instantiation', 'Level-of-detail', 'Manner', 'Substitution'])]

df.sort_values(by=['file'], ascending=True, inplace=True)
df.to_csv('Data/PDTB-3.0/pdtb_3.csv', index=False)
print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')

print(f"Preparing Lin partition of the PDTB 3.0 corpus...")
df_lin = df[df['folder'] == '23']
df_lin.to_csv('Data/PDTB-3.0/pdtb_3_lin.csv', index=False)
print(f"Done.")

print(f"Preparing Ji partition of the PDTB 3.0 corpus...")
df_ji = df[df['folder'].isin(['21', '22'])]
df_ji.to_csv('Data/PDTB-3.0/pdtb_3_ji.csv', index=False)
print(f"Done.")

print(f"Preparing balanced partition of the PDTB 3.0 corpus...")
df_no_multi = df[df['multi_sense'] == '']
gs_test = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gs_test.split(df_no_multi, df_no_multi['sense']))
df_balanced = df_no_multi.iloc[test_idx].copy()
df_balanced.sort_values(by=['file'], ascending=True, inplace=True)
df_balanced.to_csv('Data/PDTB-3.0/pdtb_3_balanced.csv', index=False)
print(f"Done.")


# !!! UNCOMMENT TO CHECK SPECIFIC DOCUMENT !!! #

# gold_path = 'Corpora/PDTB-3.0/gold'
# raw_path = 'Corpora/PDTB-3.0/raw'
# folder = '21'
# file = 'wsj_2145'
# with open(raw_path+'/'+folder+'/'+file, 'r', encoding = 'ISO-8859-1') as raw_text:
#     print(raw_text.read().rstrip())
# for line in open(gold_path+'/'+folder+'/'+file, 'r', encoding = 'ISO-8859-1').readlines():
#     # if line.startswith('Implicit'):
#     print(line.rstrip())