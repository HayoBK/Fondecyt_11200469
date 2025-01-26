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
import matplotlib.patches as mpatches
from scipy.stats import kruskal


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
trial_means = navi_filtered.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars].mean().reset_index()

trial_means = trial_means.merge(navi_filtered[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')
trial_means.rename(columns={"True_Block": "Setting"},inplace=True)
trial_means['Setting'] = trial_means['Setting'].replace({
    'HiddenTarget_1': 'Ego-Allocentric',
    'HiddenTarget_3': 'Mainly Allocentric'})

trial_means = trial_means.rename(columns={"Modalidad": "Modality"})
trial_means["Modality"] = trial_means["Modality"].replace({
    "No Inmersivo": "Non-immersive (NI)",
    "Realidad Virtual": "Virtual Reality (RV)"
})

df = trial_means.copy()


# Configurar estilo general de los gráficos
sns.set_palette(["#ADD8E6", "#FFA07A", "#98FB98"])
sns.set(style='white', font_scale=2, rc={'figure.figsize': (20, 24)})

# Crear la figura y los ejes
fig, axes = plt.subplots(2, 2, figsize=(20, 24), sharey=True, sharex=True)

# Métricas y configuraciones
#metrics = ["CSE (normalized)", "H-total (normalized)"]
#metrics = ["CSE","H-total"]

# Normalizar los datos para CSE y H-total eliminando diferencias por Modality
def normalize_within_modality(df, metric):
    normalized_col = f"{metric} (normalized)"
    df[normalized_col] = df.groupby("Modality")[metric].transform(lambda x: (x - x.mean()) / x.std())
    return df

# Normalizar las métricas
metrics_to_normalize = ["CSE", "H-total"]
for metric in metrics_to_normalize:
    df = normalize_within_modality(df, metric)
# Eliminar outliers con Z-score mayor a 3 en el grupo Vestibular
for metric in ["CSE (normalized)", "H-total (normalized)"]:
    df = df[~((df["Group"] == "Vestibular non-PPPD") & (df[metric] > 3))]
df_filtered = df[df["Modality"] != "Virtual Reality (RV)"]


settings = ["Ego-Allocentric", "Mainly Allocentric"]
modalities = ["Non-immersive (NI)", "Virtual Reality (RV)"]
hue_order = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
palette = {"PPPD": "#ADD8E6", "Vestibular non-PPPD": "#FFA07A", "Healthy Volunteer": "#98FB98"}

metrics = ["CSE (normalized)", "H-total (normalized)"]
titles = ["CSE (normalized)", "H-total (normalized)"]

Mi_Orden = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
df = df[df["Modality"] != "Virtual Reality (RV)"]

# Crear la figura y los ejes
fig, axes = plt.subplots(2, 1, figsize=(16, 20), sharey=False, sharex=True)

# Métricas y configuraciones
metrics = ["CSE (normalized)", "H-total (normalized)"]
hue_order = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
palette = {"PPPD": "#ADD8E6", "Vestibular non-PPPD": "#FFA07A", "Healthy Volunteer": "#98FB98"}

# Crear un gráfico por fila
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.boxplot(
        data=df,
        x="Setting",
        y=metric,
        hue="Group",
        hue_order=hue_order,
        ax=ax,
        palette=palette,
        linewidth=6
    )
    ax.set_title(f"{metric} by Setting", fontsize=35, fontweight='bold')
    ax.set_ylabel("Z-score (normalized)", fontsize=20, fontweight='bold')
    ax.set_xlabel("Setting", fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=28, rotation=0)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(-2.5, 3)  # Escala fija en Y entre -3 y 3
    ax.legend(title="Group", fontsize=22, loc='lower right', frameon=False)

axes[0].xaxis.set_tick_params(labelbottom=True)

for label in axes[0].get_xticklabels():
    label.set_fontweight('bold')
axes[-1].set_xlabel(" ", fontsize=28, fontweight='bold')
# Aplicar negrita a los ticks del eje X
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

# Ajustar diseño general
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Dejar espacio para la leyenda global
output_file = Output_Dir + "Figura 3 - Spatial Navigation (per Setting).png"
plt.savefig(output_file)
plt.show()


# Pruebas de diferencias entre settings para cada grupo y métrica
metrics = ["CSE (normalized)", "H-total (normalized)"]
hue_order = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
settings = df["Setting"].unique()

effect_sizes = []
print("\nResultados de Kruskal-Wallis por Grupo y Setting:")
for metric in metrics:
    print(f"\nMétrica: {metric}")
    for group in hue_order:
        subset = df[df["Group"] == group]
        groups = [subset[subset["Setting"] == setting][metric] for setting in settings]

        # Filtrar grupos vacíos
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            print(f"  Grupo: {group}, No hay suficientes datos para realizar la prueba.")
            continue

        # Prueba de Kruskal-Wallis
        stat, p = kruskal(*groups)
        print(f"  Grupo: {group}, H-statistic: {stat:.3f}, p-value: {p:.3f}")

        # Tamaño del efecto
        n = len(subset)
        eta_squared = stat / (n - 1)
        effect_sizes.append((metric, group, eta_squared))
        print(f"    Tamaño del efecto η²: {eta_squared:.3f}")

# Evaluar diferencias de tamaño de efecto entre grupos y settings mediante bootstrap
bootstrap_results = []
print("\nResultados de Bootstrap para diferencias de tamaño de efecto entre Settings:")
for metric in metrics:
    eta_diffs = []
    for _ in range(1000):  # Iteraciones de bootstrap
        resampled_data = resample(df)
        for group in hue_order:
            subset = resampled_data[resampled_data["Group"] == group]
            group_eta = []
            for setting in settings:
                group_data = subset[subset["Setting"] == setting][metric]

                # Filtrar datos insuficientes
                if len(group_data) > 1:
                    group_eta.append(group_data.mean())  # Usar la media para el cálculo de diferencias

            # Solo calcular diferencia si hay datos para ambos settings
            if len(group_eta) == 2:
                eta_diffs.append(group_eta[1] - group_eta[0])

    if eta_diffs:  # Verificar que existan diferencias calculadas
        # Calcular IC del bootstrap
        ci_lower = np.percentile(eta_diffs, 2.5)
        ci_upper = np.percentile(eta_diffs, 97.5)
        mean_diff = np.mean(eta_diffs)
        print(f"Métrica: {metric}, Diferencia promedio de η²: {mean_diff:.3f}, CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        bootstrap_results.append((metric, mean_diff, ci_lower, ci_upper))
    else:
        print(f"Métrica: {metric}, No se pudo calcular diferencias de bootstrap debido a datos insuficientes.")



p_values = {}
print("\nResultados de Kruskal-Wallis por Grupo y Setting:")
for metric in metrics:
    p_values[metric] = {}
    print(f"\nMétrica: {metric}")
    for group in hue_order:
        subset = df[df["Group"] == group]
        groups = [subset[subset["Setting"] == setting][metric] for setting in settings]

        # Filtrar grupos vacíos
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            print(f"  Grupo: {group}, No hay suficientes datos para realizar la prueba.")
            continue

        # Prueba de Kruskal-Wallis
        stat, p = kruskal(*groups)
        print(f"  Grupo: {group}, H-statistic: {stat:.3f}, p-value: {p:.3f}")
        p_values[metric][group] = p

# Variables necesarias
metrics = ["CSE (normalized)", "H-total (normalized)"]
hue_order = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
settings = ['Ego-Allocentric', 'Mainly Allocentric']

# Paleta de colores personalizada
custom_palette = [
    "#ADD8E6", "#92BCD1",  # Light Blue, Darker Blue
    "#FFA07A", "#E08A6D",  # Light Salmon, Darker Salmon
    "#98FB98", "#82D782"    # Light Green, Darker Green
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

# Crear la figura y los ejes
fig, axes = plt.subplots(2, 1, figsize=(22, 20), sharey=False, sharex=True)

# Crear los gráficos
for i, metric in enumerate(metrics):
    ax = axes[i]

    # Crear el boxplot base
    sns.boxplot(
        data=df,
        x="Group",
        order=hue_order,
        y=metric,
        hue="Setting",
        ax=ax,
        hue_order=settings,
        dodge=True,
        linewidth=4,
        fliersize=0,
        legend = False
    )

    # Aplicar colores y tramas manualmente
    patches = [patch for patch in ax.patches if
               isinstance(patch, mpatches.PathPatch)]  # Asegurarse de iterar solo sobre cajas
    for j, patch in enumerate(patches):
        # Evitar desbordamiento de índices
        group_idx = j // len(settings)  # Determinar índice del grupo
        setting_idx = j % len(settings)  # Determinar índice del setting

        # Validar índices dentro de los límites esperados
        if group_idx < len(hue_order) and setting_idx < len(settings):
            group = hue_order[group_idx]
            setting = settings[setting_idx]

            # Aplicar color y trama
            color_idx = color_mapping[(group, setting)]
            patch.set_facecolor(custom_palette[color_idx])
            patch.set_hatch(hatch_mapping[setting])

    # Títulos y etiquetas
    ax.set_title(f"{metric} by Group \n and Setting (Ego-Allocentric)", fontsize=32, fontweight='bold')
    ax.set_ylabel("Z-score (normalized)", fontsize=20, fontweight='bold')
    ax.set_xlabel("Group", fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', labelsize=28, rotation=0)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_ylim(-2.5, 3)  # Escala fija en Y entre -2.5 y 3

axes[0].xaxis.set_tick_params(labelbottom=True)

#for label in axes[0].get_xticklabels():
#    label.set_fontweight('bold')

axes[-1].set_xlabel(" ", fontsize=28, fontweight='bold')

# Añadir barras y anotaciones para p-values
for j, group in enumerate(hue_order):
    subset = df[df["Group"] == group]
    values = [subset[subset["Setting"] == setting][metric].dropna() for setting in settings]

    if len(values) == 2 and all(len(v) > 0 for v in values):
        stat, p = kruskal(*values)
        if p < 0.05:
            x1, x2 = j - 0.2, j + 0.2
            y, h, col = max(max(v) for v in values) + 0.2, 0.2, 'black'
            ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2, c=col)
            ax.text((x1 + x2) / 2, y + h, f"* (p={p:.3f})", ha='center', va='bottom', color=col, fontsize=12)
# Crear leyenda personalizada
patches = [
    mpatches.Patch(facecolor=custom_palette[color_mappingL[(group, setting)]],
                   hatch=hatch_mapping[setting],
                   linewidth=4, edgecolor='black',
                   label=f"{group} ({setting})")
    for group in hue_order for setting in settings
]
fig.legend(
    handles=patches, title="Group and Setting", loc='center right', ncol=1, fontsize=22, title_fontsize=28,
    handleheight=3, handlelength=3,
    frameon=False
)

# Ajustar diseño general
plt.tight_layout(rect=[0, 0, 0.8, 1])  # Dejar espacio para la leyenda global
output_file = Output_Dir + "Figura 3b - Spatial Navigation (per Setting).png"
plt.savefig(output_file)
plt.show()
