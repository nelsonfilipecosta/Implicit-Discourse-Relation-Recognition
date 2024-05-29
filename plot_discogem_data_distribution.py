import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('Data/DiscoGeM/discogem_train.csv')
df_validation = pd.read_csv('Data/DiscoGeM/discogem_validation.csv')
df_test = pd.read_csv('Data/DiscoGeM/discogem_test.csv')

df_total = df_train.shape[0] + df_validation.shape[0] + df_test.shape[0]
df_train_size = df_train.shape[0]
df_train_percentage = round((df_train_size / df_total) * 100)
df_validation_size = df_validation.shape[0]
df_validation_percentage = round((df_validation_size / df_total) * 100)
df_test_size = df_test.shape[0]
df_test_percentage = round((df_test_size / df_total) * 100)

plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

figure, (axis_1, axis_2, axis_3) = plt.subplots(1, 3, figsize=(24, 8))

v_offset = 0.966
h_offset = 0.974

axis_1.set_title(f'Train Split', fontweight='bold')
axis_1.tick_params(axis='x', rotation=90)
axis_1.set_ylim((0, 1500))
axis_1.set_yticks(np.arange(0, 1501, 500))
axis_1.bar(df_train['majority_level_3'].value_counts().index, df_train['majority_level_3'].value_counts(), color='tab:green')
axis_1.text(h_offset, v_offset, 'Total Implicit DR:  ' + str(df_train.shape[0]) + ' (' + str(df_train_percentage) + '%)',
            ha='right', va='top', transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', alpha=0.1))

axis_2.set_title(f'Validation Split', fontweight='bold')
axis_2.tick_params(axis='x', rotation=90)
axis_2.set_ylim((0, 225))
axis_2.set_yticks(np.arange(0, 226, 75))
axis_2.bar(df_validation['majority_level_3'].value_counts().index, df_validation['majority_level_3'].value_counts(), color='tab:red')
axis_2.text(h_offset, v_offset, 'Total Implicit DR:  ' + str(df_validation.shape[0]) + ' (' + str(df_validation_percentage) + '%)',
             ha='right', va='top', transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:red', alpha=0.1))

axis_3.set_title(f'Test Split', fontweight='bold')
axis_3.tick_params(axis='x', rotation=90)
axis_3.set_ylim((0, 400))
axis_3.set_yticks(np.arange(0, 451, 150))
axis_3.bar(df_test['majority_level_3'].value_counts().index, df_test['majority_level_3'].value_counts(), color='tab:blue')
axis_3.text(h_offset, v_offset, 'Total Implicit DR:  ' + str(df_test.shape[0]) + ' (' + str(df_test_percentage) + '%)',
            ha='right', va='top', transform=axis_3.transAxes, bbox=dict(boxstyle='square', facecolor='tab:blue', alpha=0.1))

plt.tight_layout()
figure.savefig('Data/DiscoGeM/discogem_label_distribution.png', format='png', dpi=300, bbox_inches="tight")