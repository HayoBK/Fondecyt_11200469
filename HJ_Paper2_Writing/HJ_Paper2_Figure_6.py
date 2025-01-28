# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 28, Martes.
# Vamos de lleno a ESCRIBIR el Paper Numero 2. - Figura 6
# Mejor un Script por figura, sino yo y ChatGPT empiezan a colapsar. Este está listo.
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from scipy.stats import friedmanchisquare
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.utils import resample
import numpy as np
import matplotlib.patches as mpatches
from scipy.stats import kruskal


Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figura 6/")

file_path = Py_Processing_Dir + "A_INFINITE_BASAL_DF.xlsx"
codex_df = pd.read_excel(file_path)
file_path = Py_Processing_Dir + "F_Fixations_4All.csv"
fix_df = pd.read_csv(file_path)
file_path = Py_Processing_Dir + "H_SimianMaze_ShortDf_Normalized.csv"
navi_df = pd.read_csv(file_path)

# Renombrar columnas según los nuevos nombres
navi_df.rename(columns={
    "Latencia": "Latency",
    "Indice_Eficiencia": "Efficiency Index",
    "Herror": "H-error",
    "Hpath": "H-path",
    "Htotal": "H-total"
}, inplace=True)

navi_df.rename(columns={
    "Latencia_norm": "Latency (normalized)",
    "Indice_Eficiencia_norm": "Efficiency Index (normalized) ",
    "Herror_norm": "H-error (normalized)",
    "Hpath_norm": "H-path (normalized)",
    "Htotal_norm": "H-total (normalized)",
    "Path_length_norm": "Path length (normalized)",
    "CSE_norm": "CSE (normalized)"
}, inplace=True)
# Renombrar grupos
navi_df['Grupo'] = navi_df['Grupo'].replace({
    'MPPP': 'PPPD',
    'Vestibular': 'Vestibular non-PPPD',
    'Voluntario Sano': 'Healthy Volunteer'
})

navi_df.rename(columns={"Grupo": "Group"}, inplace=True)

output_vars = ["CSE","H-total"]
output_vars_norm = ["H-error (normalized)", "H-path (normalized)","H-total (normalized)"]
"""
Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_2','HiddenTarget_3']
navi_filtered = navi_df[navi_df['True_Block'].isin(Interesting_blocks)].copy()

# Promediar cada parámetro por trial dentro de cada True_Block
trial_means = navi_filtered.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars].mean().reset_index()

trial_means = trial_means.merge(navi_filtered[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')
trial_means.rename(columns={"True_Block": "Setting"},inplace=True)
trial_means['Setting'] = trial_means['Setting'].replace({
    'HiddenTarget_1': 'Ego-Allocentric',
    'HiddenTarget_3': 'Mainly Allocentric'})
"""
# Filtrar y fusionar bloques interesantes
Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
navi_filtered = navi_df[navi_df['True_Block'].isin(Interesting_blocks)].copy()
min_z = 4
navi_filtered['Entropy Ratio'] = (navi_filtered['H-path (normalized)'] - min_z) / (navi_filtered['H-total (normalized)'] - min_z)
output_vars_norm = ["H-error (normalized)", "H-path (normalized)","H-total (normalized)","Entropy Ratio"]

# Fusionar HiddenTarget_1 y HiddenTarget_2 promediando valores para cada combinación de sujeto y modalidad
merged_block = navi_filtered[navi_filtered['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2'])]
merged_block_means = merged_block.groupby(['Sujeto', 'Modalidad'])[output_vars_norm].mean().reset_index()
merged_block_means['True_Block'] = 'HiddenTarget_1_2'

# Filtrar HiddenTarget_3
remaining_block = navi_filtered[navi_filtered['True_Block'] == 'HiddenTarget_3']

# Concatenar bloques fusionados con el bloque restante
navi_filtered = pd.concat([merged_block_means, remaining_block], ignore_index=True)

# Promediar cada parámetro por trial dentro de cada True_Block
trial_means = navi_filtered.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars_norm].mean().reset_index()

trial_means = trial_means.merge(navi_filtered[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')
trial_means.rename(columns={"True_Block": "Setting"}, inplace=True)
trial_means['Setting'] = trial_means['Setting'].replace({
    'HiddenTarget_1_2': 'Ego-Allocentric',
    'HiddenTarget_3': 'Mainly Allocentric'
})

trial_means = trial_means.rename(columns={"Modalidad": "Modality"})
trial_means["Modality"] = trial_means["Modality"].replace({
    "No Inmersivo": "Non-immersive (NI)",
    "Realidad Virtual": "Virtual Reality (RV)"
})

df = trial_means.copy()
df = df[df['Group'].notna()]
df= df[df['Modality'] == "Non-immersive (NI)"]


# Configuración del tamaño del gráfico
plt.figure(figsize=(10, 6))

# Crear el boxplot
sns.boxplot(
    data=df,
    x='Group',
    y='Entropy Ratio',
    showfliers=True,  # Muestra outliers (puedes cambiar a False si no los quieres)
    palette='Set2'    # Paleta de colores (ajústala si prefieres otra)
)

# Personalizar las etiquetas
plt.title('Boxplot of Entropy Ratio by Group and Setting', fontsize=14)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Entropy Ratio', fontsize=12)

# Mostrar la leyenda fuera del gráfico
plt.legend(title='Setting', bbox_to_anchor=(1.05, 1), loc='upper left')

# Mostrar el gráfico
plt.tight_layout()
plt.show()