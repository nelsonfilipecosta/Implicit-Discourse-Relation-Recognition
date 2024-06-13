import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/PDTB-3.0/pdtb_3.csv')
df_ji = pd.read_csv('Data/PDTB-3.0/pdtb_3_ji.csv')
df_lin = pd.read_csv('Data/PDTB-3.0/pdtb_3_lin.csv')
df_balanced = pd.read_csv('Data/PDTB-3.0/pdtb_3_balanced.csv')

plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

figure, (axis_1, axis_2, axis_3) = plt.subplots(1, 3, figsize=(24, 8))

v_offset = 0.966
h_offset = 0.974

axis_1.set_title(f'Lin Test Split', fontweight='bold')
axis_1.tick_params(axis='x', rotation=90)
axis_1.set_ylim((0, 300))
axis_1.bar(df_lin['sense2'].value_counts().index, df_lin['sense2'].value_counts(), color='tab:green')
axis_1.text(h_offset, v_offset, 'Total Implicit DR:   995 \n One-Sense Implitcit DR:     959 \n Two-Senses Implicit DR:       36 ',
            ha='right', va='top', transform=axis_1.transAxes, bbox=dict(boxstyle='square', facecolor='tab:green', alpha=0.1))

axis_2.set_title(f'Ji Test Split', fontweight='bold')
axis_2.tick_params(axis='x', rotation=90)
axis_2.set_ylim((0, 450))
axis_2.set_yticks(np.arange(0, 451, 75))
axis_2.bar(df_ji['sense2'].value_counts().index, df_ji['sense2'].value_counts(), color='tab:red')
axis_2.text(h_offset, v_offset, 'Total Implicit DR:  1,453 \n One-Sense Implitcit DR:  1,387 \n Two-Senses Implicit DR:       66 ',
             ha='right', va='top', transform=axis_2.transAxes, bbox=dict(boxstyle='square', facecolor='tab:red', alpha=0.1))

axis_3.set_title(f'Balanced Test Split', fontweight='bold')
axis_3.tick_params(axis='x', rotation=90)
axis_3.set_ylim((0, 1200))
axis_3.set_yticks(np.arange(0, 1201, 150))
axis_3.bar(df_balanced['sense2'].value_counts().index, df_balanced['sense2'].value_counts(), color='tab:blue')
axis_3.text(h_offset, v_offset, 'Total Implicit DR:  4,102 \n One-Sense Implitcit DR:  4,102 \n Two-Senses Implicit DR:         0 ',
            ha='right', va='top', transform=axis_3.transAxes, bbox=dict(boxstyle='square', facecolor='tab:blue', alpha=0.1))

plt.tight_layout()
figure.savefig('Data/PDTB-3.0/pdtb_label_distribution.png', format='png', dpi=300, bbox_inches="tight")