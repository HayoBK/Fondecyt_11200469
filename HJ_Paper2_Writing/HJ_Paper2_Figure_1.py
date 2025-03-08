# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 21, Martes.
# Vamos de lleno a ESCRIBIR el Paper Numero 2. - Figura 1
# Mejor un Script por figura, sino yo y ChatGPT empiezan a colapsar. Este está listo.
# -----------------------------------------------------------------------
# %%
import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figura 1/")

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


output_vars = ["Path_length", "Efficiency Index", "Latency", "CSE", "H-error", "H-path", "H-total"]
output_vars_norm = ["Path_length_norm", "Efficiency Index_norm", "Latency_norm", "CSE_norm", "H-error_norm",
                    "H-path_norm", "H-total_norm"]
# -----------------------------------------------------------------------
# FIGURA 1
# -----------------------------------------------------------------------
#navi_df = navi_df[navi_df['Sujeto'].str.extract('(\d+)', expand=False).astype(int) >= 30]

# Filtrar datos para solo analizar bloques interesantes
Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_3']
navi_filtered = navi_df[navi_df['True_Block'].isin(Interesting_blocks)].copy()

# Convertir Latencia de milisegundos a segundos
navi_filtered["Latency"] = navi_filtered["Latency"] / 1000

# Promediar cada parámetro por trial dentro de cada True_Block
trial_means = navi_filtered.groupby(['Sujeto', 'Modalidad', 'True_Block'])[output_vars].mean().reset_index()

# Promediar los bloques para cada Modalidad y Sujeto
block_means = trial_means.groupby(['Sujeto', 'Modalidad'])[output_vars].mean().reset_index()

# Promediar las modalidades para cada Sujeto
subject_means = block_means.groupby('Sujeto')[output_vars].mean().reset_index()

# Agregar la información del grupo a los datos
subject_means = subject_means.merge(navi_filtered[['Sujeto', 'Group']].drop_duplicates(), on='Sujeto', how='left')

# Generar una tabla con los valores promedio por Grupo
print("Generando tabla de valores promedio por Grupo...")
group_means = subject_means.groupby('Group')[output_vars].mean().reset_index()
output_table_file = Output_Dir + "Resumen_Promedios_Grupos.csv"
group_means.to_csv(output_table_file, index=False)

print("Tabla de valores promedio por Grupo:")
print(group_means)
print(f"Tabla guardada en: {output_table_file}")

# Realizar prueba de Kruskal-Wallis y post-hoc de Dunn
print("Realizando pruebas de Kruskal-Wallis y post-hoc de Dunn...")
kruskal_results = []
posthoc_results = {}

for var in output_vars:
    # Preparar datos por grupo
    data_by_group = [
        subject_means[subject_means['Group'] == grupo][var].dropna()
        for grupo in subject_means['Group'].unique()
    ]

    # Prueba de Kruskal-Wallis
    stat, p_value = kruskal(*data_by_group)
    kruskal_results.append({
        "Variable": var,
        "Kruskal-Wallis Statistic": stat,
        "p-Value": p_value
    })

    # Post-hoc de Dunn si p < 0.05
    if p_value < 0.05:
        posthoc = posthoc_dunn(subject_means, val_col=var, group_col='Group', p_adjust='bonferroni')
        posthoc_results[var] = posthoc

# Guardar resultados de Kruskal-Wallis
kruskal_results_df = pd.DataFrame(kruskal_results)
kruskal_file = Output_Dir + "Kruskal_Wallis_Results.csv"
kruskal_results_df.to_csv(kruskal_file, index=False)
print(f"Resultados de Kruskal-Wallis guardados en: {kruskal_file}")

# Guardar resultados post-hoc
for var, posthoc in posthoc_results.items():
    posthoc_file = Output_Dir + f"Posthoc_Dunn_{var}.csv"
    posthoc.to_csv(posthoc_file, index=True)
    print(f"Resultados post-hoc de Dunn para {var} guardados en: {posthoc_file}")

# Configurar estilo de los gráficos
sns.set_palette(["#ADD8E6", "#FFA07A", "#98FB98"])
sns.set(style='white', font_scale=2, rc={'figure.figsize': (30, 20)})

Mi_Orden = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']

# Generar una grilla de gráficos para comparar los grupos
print("Generando grilla de gráficos para las variables...")
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(30, 20))
axes = axes.flatten()

# Etiquetas para el eje Y
y_labels = {
    "Path_length": "Pool diameters",
    "Efficiency Index": "Efficiency index",
    "Latency": "Seconds",
    "CSE": "CSE",
    "H-error": "Entropy",
    "H-path": "Entropy",
    "H-total": "Entropy"
}

# Obtener p-values para anotarlos en los gráficos
p_values = {result['Variable']: result['p-Value'] for result in kruskal_results}
stat_p_values = {result['Variable']: (result['Kruskal-Wallis Statistic'], result['p-Value']) for result in kruskal_results}

# Añadir leyenda manualmente en el espacio inferior derecho
# Definir leyenda manualmente con diseño en columna y marco gris
legend_handles = [
    plt.Line2D([0], [0], color="#ADD8E6", lw=20, label="PPPD", markeredgecolor='gray', markeredgewidth=9.5),
    plt.Line2D([0], [0], color="#FFA07A", lw=6, label="Vestibular non-PPPD", markeredgecolor='gray', markeredgewidth=1.5),
    plt.Line2D([0], [0], color="#98FB98", lw=6, label="Healthy Volunteer", markeredgecolor='gray', markeredgewidth=1.5)
]

legend_handles = [
    Patch(facecolor="#ADD8E6", edgecolor='gray', linewidth=4, label="PPPD"),
    Patch(facecolor="#FFA07A", edgecolor='gray', linewidth=4, label="Vestibular non-PPPD"),
    Patch(facecolor="#98FB98", edgecolor='gray', linewidth=4, label="Healthy Volunteer")
]

for i, var in enumerate(output_vars):
    if i < len(axes):
        ax = sns.boxplot(
            data=subject_means,
            x="Group",
            y=var,
            linewidth=6,
            order=Mi_Orden,
            hue="Group",
            palette={"PPPD": "#ADD8E6", "Vestibular non-PPPD": "#FFA07A", "Healthy Volunteer": "#98FB98"},
            ax=axes[i],
            showfliers=False
        )
        sns.stripplot(
            data=subject_means,
            x="Group",
            y=var,
            jitter=True,
            color='black',
            dodge=False,
            size=10,
            ax=ax,
            order=Mi_Orden
        )

        # Ajustar etiquetas del eje X
        ax.set_xticks(range(len(Mi_Orden)))
        ax.set_xticklabels(['PPPD', 'Vestibular\nnon-PPPD', 'Healthy\nVolunteer'], rotation=0, fontsize=20, linespacing=1.5)

        # Añadir título y ajustar estilo
        ax.set_title(f"{var}", fontsize=32, fontweight='bold')
        ax.set_xlabel(" ")
        ax.set_ylabel(y_labels.get(var, var), weight='bold')

        stat, p_value = stat_p_values[var]
        annotation_text = f"H = {stat:.2f}, p = {p_value:.3f}"
        ax.text(0.95, 0.97, annotation_text, transform=ax.transAxes, fontsize=20,
                verticalalignment='top', horizontalalignment='right', fontweight='bold')

# Añadir leyenda manualmente al espacio inferior derecho
fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.95, 0.2), title="Groups", fontsize=30, ncol=1, frameon=False)

# Eliminar cualquier gráfico vacío
for j in range(len(output_vars), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_file = Output_Dir + "Figura 1 - Spatial Navigation (whole).png"
plt.savefig(output_file)
plt.show()

print(f"Grilla de gráficos guardada en: {output_file}")