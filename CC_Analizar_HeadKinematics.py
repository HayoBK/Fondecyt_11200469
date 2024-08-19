#-------------------------------------------
#
#   Mayo 13, 2024.
#   Vamos a revisar los headKinematics.
#
#-------------------------------------------
import pandas as pd

import os

import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
import numpy as np
from pathlib import Path
import socket
import scipy.stats as stats
import scikit_posthocs as sp

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

nombre_host = socket.gethostname()
print(nombre_host)
if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

final_summarized_df = pd.read_csv(Py_Processing_Dir+'CB_HeadKinematics.csv')

filtered_df = final_summarized_df[
    (final_summarized_df['Modality'] == 'Realidad Virtual') &
    (final_summarized_df['MWM_Bloque'].isin(['HiddenTarget_1']))
]



# Agrupar por 'Subject' y calcular el promedio de las desviaciones estándar de las variables de interés
summary_df = filtered_df.groupby(['Subject', 'MWM_Bloque']).agg({
    'vX_std': 'mean',
    'vY_std': 'mean',
    'vZ_std': 'mean',
    'vRoll_std': 'mean',
    'vJaw_std': 'mean',
    'vPitch_std': 'mean',
    'Grupo': 'first'  # Asumiendo que todos los registros por subject tienen el mismo grupo
}).reset_index()

print(summary_df)

sns.set(style="whitegrid")
summary_df=filtered_df
# Crear una figura con subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 18))  # Ajusta el tamaño según necesidades
axes = axes.flatten()  # Para iterar fácilmente en un loop

variables = ['vX_std', 'vY_std', 'vZ_std', 'vRoll_std', 'vJaw_std', 'vPitch_std']
for i, var in enumerate(variables):
    sns.boxplot(x='Grupo', y=var, data=summary_df, ax=axes[i])
    axes[i].set_title(f'Boxplot of {var} by Group')
    axes[i].set_xlabel('Group')
    axes[i].set_ylabel(var)

plt.tight_layout()
plt.show()


results = {}
for var in variables:
    groups = [group[var].dropna() for name, group in summary_df.groupby('Grupo')]
    kruskal_result = stats.kruskal(*groups)
    results[var] = kruskal_result

    print(f"Kruskal-Wallis test for {var}:")
    print("Statistic:", kruskal_result.statistic, "p-value:", kruskal_result.pvalue, "\n")

sns.boxplot(x='MWM_Bloque', y='vJaw_std', hue='Grupo', data=summary_df)
plt.title('Boxplot of vJaw_std by MWM_Bloque and Group')
plt.xlabel('MWM_Bloque')
plt.ylabel('vJaw_std')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.show()

sns.boxplot(x='Grupo', y='vJaw_std', data=summary_df)
plt.title('Boxplot of vJaw_std by  Group')
plt.xlabel('Grupo')
plt.ylabel('vJaw_std')
plt.legend(title='Group')
plt.show()

overall_stat, overall_p = stats.kruskal(*[group['vJaw_std'].dropna() for name, group in summary_df.groupby('Grupo')])
print(f"Overall Kruskal-Wallis test across all MWM_Bloques: Statistic = {overall_stat}, p-value = {overall_p}")

data_groups = [group['vJaw_std'].dropna() for name, group in summary_df.groupby('Grupo')]

# Perform Dunn's test for multiple comparisons
p_values = sp.posthoc_dunn(data_groups, p_adjust='bonferroni')  # 'bonferroni' adjustment for multiple comparisons

print(p_values)


# Ensure 'Grupo' and 'vJaw_std' are in summary_df and 'vJaw_std' has valid data
summary_df2 = summary_df.dropna(subset=['vJaw_std'])

# Perform the Dunn's test using the DataFrame directly
if not summary_df2.empty and summary_df2['Grupo'].nunique() > 1:
    p_values = sp.posthoc_dunn(summary_df2, val_col='vJaw_std', group_col='Grupo', p_adjust='bonferroni')
    print(p_values)
else:
    print("Insufficient data or groups for statistical testing.")

# Kruskal-Wallis test within each MWM_Bloque
results = {}
for bloque in summary_df['MWM_Bloque'].unique():
    bloque_data = summary_df[summary_df['MWM_Bloque'] == bloque]
    kruskal_result = stats.kruskal(*[group['vJaw_std'].dropna() for name, group in bloque_data.groupby('Grupo')])
    results[bloque] = kruskal_result
    print(f"Kruskal-Wallis test for {bloque}: Statistic = {kruskal_result.statistic}, p-value = {kruskal_result.pvalue}")