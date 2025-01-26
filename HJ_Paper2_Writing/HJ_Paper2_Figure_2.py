# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 25, Sábado.
# Vamos de lleno a ESCRIBIR el Paper Numero 2. - Figura 2
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

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figura 3/")

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
output_vars_norm = ["CSE (normalized)", "H-total (normalized)"]

Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_3']
navi_filtered = navi_df[navi_df['True_Block'].isin(Interesting_blocks)].copy()

# Promediar cada parámetro por trial dentro de cada True_Block
trial_means = navi_filtered.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars_norm].mean().reset_index()

# Promediar los bloques para cada Modalidad y Sujeto
block_means = trial_means.groupby(['Sujeto', 'Modalidad'])[output_vars_norm].mean().reset_index()

block_means = block_means.merge(navi_filtered[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')

# DESDE AQUI con CHAT GPT nuevo

# Renombrar columna y valores para mayor claridad
block_means = block_means.rename(columns={"Modalidad": "Modality"})
block_means["Modality"] = block_means["Modality"].replace({
    "No Inmersivo": "Non-immersive (NI)",
    "Realidad Virtual": "Virtual Reality (RV)"
})

# Configurar estilo general de los gráficos
sns.set_palette(["#ADD8E6", "#FFA07A", "#98FB98"])
sns.set(style='white', font_scale=2, rc={'figure.figsize': (16, 20)})

# Crear la figura y los ejes
fig, axes = plt.subplots(2, 1, figsize=(16, 20), sharex=True)
metrics = ["CSE (normalized)", "H-total (normalized)"]
titles = ["CSE (normalized)", "H-total (normalized)"]

Mi_Orden = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']


for ax, metric, title in zip(axes, metrics, titles):
    sns.boxplot(
        data=block_means,
        x="Modality",
        y=metric,
        hue_order=Mi_Orden,
        linewidth=6,
        hue="Group",
        ax=ax,
        palette={"PPPD": "#ADD8E6", "Vestibular non-PPPD": "#FFA07A", "Healthy Volunteer": "#98FB98"}# Puedes ajustar el esquema de colores si lo prefieres
    )
    ax.set_title(title, fontsize=30, fontweight='bold')
    ax.set_ylabel("Z-score (normalized)", fontsize=24, fontweight='bold')
    #ax.set_xlabel("")  # Dejar vacío para gráficos superiores
    ax.legend(title="Group", fontsize=22, loc='upper right', frameon=False)
    ax.tick_params(axis='x', labelsize=27, bottom=True)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')  # Aplicar negritafor label in axes[0].get_xticklabels():
# Ajustar eje X para el gráfico inferior
axes[0].xaxis.set_tick_params(labelbottom=True)

for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')
axes[-1].set_xlabel(" ", fontsize=28, fontweight='bold')

# Ajustar diseño general
plt.tight_layout()
output_file = Output_Dir + "Figura 2 - Spatial Navigation (per Modality).png"
plt.savefig(output_file)
plt.show()
#%%
for metric in metrics:
    print(f"\nResultados de Kruskal-Wallis para {metric}:")

    for modality in block_means["Modality"].unique():
        subset = block_means[block_means["Modality"] == modality]
        groups = [subset[subset["Group"] == g][metric] for g in Mi_Orden]

        # Prueba de Kruskal-Wallis
        stat, p = kruskal(*groups)
        print(f"  Modality: {modality}, H-statistic: {stat:.3f}, p-value: {p:.3f}")

        # Conclusión
        if p > 0.05:
            print(f"    No hay diferencias significativas entre grupos (p={p:.3f})")
        else:
            print(f"    Hay diferencias significativas entre grupos (p={p:.3f})")


effect_sizes = []

for metric in metrics:
    print(f"\nTamaño del efecto Kruskal-Wallis para {metric}:")

    for modality in block_means["Modality"].unique():
        subset = block_means[block_means["Modality"] == modality]
        groups = [subset[subset["Group"] == g][metric] for g in Mi_Orden]

        # Prueba de Kruskal-Wallis
        stat, p = kruskal(*groups)
        n = len(subset)
        eta_squared = stat / (n - 1)

        print(f"  Modality: {modality}, H-statistic: {stat:.3f}, p-value: {p:.3f}, η²: {eta_squared:.3f}")
        effect_sizes.append((metric, modality, eta_squared))

# Iterar por cada métrica
block_means_renamed = block_means.rename(columns={
    "CSE (normalized)": "CSE_normalized",
    "H-total (normalized)": "H_total_normalized"
})


# Iterar por cada métrica
for metric in ["CSE_normalized", "H_total_normalized"]:
    print(f"\nANOVA para {metric.replace('_', ' ')}:")

    # Modelo ANOVA de dos vías
    model = ols(f'{metric} ~ C(Group) * C(Modality)', data=block_means_renamed).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Calcular tamaño del efecto parcial η²
    interaction_ss = anova_table.loc['C(Group):C(Modality)', 'sum_sq']
    total_ss = anova_table['sum_sq'].sum()
    partial_eta_squared = interaction_ss / total_ss
    print(f"  Partial η² (interacción): {partial_eta_squared:.3f}")

bootstrap_results = []

for metric in ["CSE_normalized", "H_total_normalized"]:
    print(f"\nBootstrap para diferencias de tamaño de efecto en {metric.replace('_', ' ')}:")

    # Bootstrapping para cada modalidad
    eta_diffs = []
    modalities = block_means_renamed["Modality"].unique()
    for _ in range(1000):  # Iteraciones de bootstrap
        resampled_data = resample(block_means_renamed)

        # Calcular tamaños del efecto para cada modalidad
        eta_values = []
        for modality in modalities:
            subset = resampled_data[resampled_data["Modality"] == modality]
            groups = [subset[subset["Group"] == g][metric] for g in Mi_Orden]
            stat, _ = kruskal(*groups)
            eta_squared = stat / (len(subset) - 1)
            eta_values.append(eta_squared)

        # Guardar diferencia entre modalidades
        eta_diffs.append(eta_values[1] - eta_values[0])

    # Calcular IC del bootstrap
    ci_lower = np.percentile(eta_diffs, 2.5)
    ci_upper = np.percentile(eta_diffs, 97.5)
    mean_diff = np.mean(eta_diffs)
    print(f"  Diferencia promedio de η²: {mean_diff:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    bootstrap_results.append((metric, mean_diff, ci_lower, ci_upper))

