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

import scipy.stats as stats
import matplotlib.patches as mpatches

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
output_vars_norm = ["H-error", "H-path", "H-total", "H-error (normalized)", "H-path (normalized)","H-total (normalized)"]
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
navi_filtered['Entropy-Ratio'] = (navi_filtered['H-error'] - min_z) / (navi_filtered['H-total'] - min_z)
navi_filtered['Entropy Ratio'] = (navi_filtered['H-error (normalized)'] - min_z) / (navi_filtered['H-total (normalized)'] - min_z)

output_vars_norm = ["H-error", "H-path", "H-total", "H-error (normalized)", "H-path (normalized)","H-total (normalized)", "Entropy-Ratio", "Entropy Ratio"]

# Fusionar HiddenTarget_1 y HiddenTarget_2 promediando valores para cada combinación de sujeto y modalidad
merged_block = navi_filtered[navi_filtered['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2'])]
merged_block_means = merged_block.groupby(['Sujeto', 'Modalidad'])[output_vars_norm].mean().reset_index()
merged_block_means['True_Block'] = 'HiddenTarget_1_2'

# Filtrar HiddenTarget_3
remaining_block = navi_filtered[navi_filtered['True_Block'] == 'HiddenTarget_3']

# Concatenar bloques fusionados con el bloque restante
navi_simplified = pd.concat([merged_block_means, remaining_block], ignore_index=True)

# Promediar cada parámetro por trial dentro de cada True_Block
trial_means = navi_simplified.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars_norm].mean().reset_index()

trial_means = trial_means.merge(navi_simplified[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')
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
df = df[~((df["Group"] == "Vestibular non-PPPD") & (df["Entropy-Ratio"] > 0.59))]
# Definir orden de grupos y configuraciones
hue_order = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
settings = ['Ego-Allocentric', 'Mainly Allocentric']

# Paleta de colores personalizada
custom_palette = [
    "#ADD8E6", "#92BCD1",  # Light Blue, Darker Blue
    "#FFA07A", "#E08A6D",  # Light Salmon, Darker Salmon
    "#98FB98", "#82D782"  # Light Green, Darker Green
]

# Mapeo manual de colores y tramas
color_mapping = {
    ("PPPD", "Ego-Allocentric"): 0,
    ("PPPD", "Mainly Allocentric"): 2,
    ("Vestibular non-PPPD", "Ego-Allocentric"): 4,
    ("Vestibular non-PPPD", "Mainly Allocentric"): 1,
    ("Healthy Volunteer", "Ego-Allocentric"): 3,
    ("Healthy Volunteer", "Mainly Allocentric"): 5
}

color_mappingL = {
    ("PPPD", "Ego-Allocentric"): 0,
    ("PPPD", "Mainly Allocentric"): 1,
    ("Vestibular non-PPPD", "Ego-Allocentric"): 2,
    ("Vestibular non-PPPD", "Mainly Allocentric"): 3,
    ("Healthy Volunteer", "Ego-Allocentric"): 4,
    ("Healthy Volunteer", "Mainly Allocentric"): 5
}

hatch_mapping = {
    "Ego-Allocentric": "",
    "Mainly Allocentric": "//"
}

# Configuración del tamaño del gráfico
fig, ax = plt.subplots(figsize=(10, 6))

# Crear el boxplot con el mismo estilo de la Figura 3
sns.boxplot(
    data=df,
    x='Group',
    y='Entropy-Ratio',
    order=hue_order,
    hue='Setting',
    hue_order=settings,
    showfliers=True,
    width=0.6,
    dodge=True,
    boxprops={'edgecolor': 'black'},
    medianprops={'color': 'black'},
    linewidth=4,
    fliersize=0,
    ax=ax,
    legend=False
)
ax.set_ylim(0.45, 0.75)  # Escala fija en Y entre -2.5 y 3

# Aplicar colores y tramas manualmente
patches = [patch for patch in ax.patches if isinstance(patch, mpatches.PathPatch)]
for j, patch in enumerate(patches):
    group_idx = j // len(settings)
    setting_idx = j % len(settings)

    if group_idx < len(hue_order) and setting_idx < len(settings):
        group = hue_order[group_idx]
        setting = settings[setting_idx]

        patch.set_facecolor(custom_palette[color_mapping[(group, setting)]])
        patch.set_hatch(hatch_mapping[setting])

# Realizar pruebas estadísticas entre grupos dentro de cada Setting
results = []
for setting in settings:
    subset = df[df['Setting'] == setting]
    groups = [group['Entropy-Ratio'].dropna().values for name, group in subset.groupby('Group')]

    if len(groups) > 1:
        stat, p = stats.kruskal(*groups)
        results.append((setting, stat, p))

# Imprimir resultados estadísticos
for setting, stat, p in results:
    print(f"Kruskal-Wallis para Setting {setting}: H={stat:.3f}, p={p:.4f}")

# Añadir barras y anotaciones para p-values
for j, group in enumerate(hue_order):
    subset = df[df["Group"] == group]
    values = [subset[subset["Setting"] == setting]['Entropy-Ratio'].dropna() for setting in settings]

    if len(values) == 2 and all(len(v) > 0 for v in values):
        stat, p = stats.kruskal(*values)
        if p < 0.05:
            print(' H clave value ', stat)
            x1, x2 = j - 0.2, j + 0.2
            y, h, col = 0.68, 0.005, 'black'  # Ajustado para el nuevo ylim
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2, c=col)
            ax.text((x1 + x2) / 2, y + h, f"* (p={p:.3f})", ha='center', va='bottom', color=col, fontsize=12)

# Personalizar las etiquetas
title_fontsize =16
label_fontsize = 16
ax.tick_params(axis='x', labelsize=12, rotation=0)
plt.title('Entropy Ratio (H-error / H-total)', fontsize=title_fontsize)
plt.xlabel('Group', fontsize=label_fontsize)
plt.ylabel('Entropy Ratio', fontsize=label_fontsize)

# Crear leyenda personalizada
legend_patches = [
    mpatches.Patch(facecolor=custom_palette[color_mappingL[(group, setting)]],
                   hatch=hatch_mapping[setting],
                   linewidth=4, edgecolor='black',
                   label=f"{group} ({setting})")
    for group in hue_order for setting in settings
]
fig.legend(
    handles=legend_patches, title="Group and Setting", loc='center right', ncol=1, fontsize=12, title_fontsize=14,
    handleheight=2.2, handlelength=2.2,
    frameon=False
)

# Ajustar diseño y mostrar
plt.tight_layout(rect=[0, 0, 0.6, 1])
output_file = Output_Dir + "Figura 6.png"
plt.savefig(output_file)
plt.show()
