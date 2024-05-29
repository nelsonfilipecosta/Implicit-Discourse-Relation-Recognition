import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/PDTB-3.0/pdtb_3.csv')
df_ji = pd.read_csv('Data/PDTB-3.0/pdtb_3_ji.csv')
df_lin = pd.read_csv('Data/PDTB-3.0/pdtb_3_lin.csv')
df_balanced = pd.read_csv('Data/PDTB-3.0/pdtb_3_balanced.csv')

plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'

figure, ((axis_1, axis_2), (axis_3, axis_4)) = plt.subplots(2, 2, figsize=(24, 16))

v_offset = 0.96
h_offset = 0.984

axis_1.set_title(f'PDTB 3.0', fontweight='bold')
axis_1.tick_params(axis='x', rotation=90)
axis_1.set_ylim((0, 4500))
axis_1.set_yticks(np.arange(0, 4501, 500))
axis_1.bar(df['sense_final'].value_counts().index, df['sense_final'].value_counts(), color='tab:green')
axis_1.text(h_offset, v_offset, 'Total Implicit DR:  21,730 \n One-Sense Implitcit DR:  20,658 \n Two-Senses Implicit DR:    1,072 ',
            ha='right', va='top', transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', alpha=0.1))

axis_2.set_title(f'Lin Test Split', fontweight='bold')
axis_2.tick_params(axis='x', rotation=90)
axis_2.set_ylim((0, 200))
axis_2.bar(df_lin['sense_final'].value_counts().index, df_lin['sense_final'].value_counts(), color='tab:red')
axis_2.text(h_offset, v_offset, 'Total Implicit DR:  1,004 \n One-Sense Implitcit DR:     968 \n Two-Senses Implicit DR:       36 ',
            ha='right', va='top', transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:red', alpha=0.1))

axis_3.set_title(f'Ji Test Split', fontweight='bold')
axis_3.tick_params(axis='x', rotation=90)
axis_3.set_ylim((0, 250))
axis_3.bar(df_ji['sense_final'].value_counts().index, df_ji['sense_final'].value_counts(), color='tab:blue')
axis_3.text(h_offset, v_offset, 'Total Implicit DR:  1,466 \n One-Sense Implitcit DR:  1,399 \n Two-Senses Implicit DR:       67 ',
             ha='right', va='top', transform=axis_3.transAxes, bbox=dict(boxstyle='square', facecolor='tab:blue', alpha=0.1))

axis_4.set_title(f'Balanced Test Split', fontweight='bold')
axis_4.tick_params(axis='x', rotation=90)
axis_4.set_ylim((0, 900))
axis_4.set_yticks(np.arange(0, 901, 150))
axis_4.bar(df_balanced['sense_final'].value_counts().index, df_balanced['sense_final'].value_counts(), color='tab:orange')
axis_4.text(h_offset, v_offset, 'Total Implicit DR:  4,132 \n One-Sense Implitcit DR:  4,132 \n Two-Senses Implicit DR:         0 ',
            ha='right', va='top', transform=axis_4.transAxes, bbox=dict(boxstyle='square', facecolor='tab:orange', alpha=0.1))

plt.tight_layout()
figure.savefig('Data/PDTB-3.0/pdtb_label_distribution.png', format='png', dpi=300, bbox_inches="tight")