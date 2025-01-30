#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
import HA_ModuloArchivos as H_Mod

# ------------------------------------------------------------
# Identificar primero en qué computador estamos trabajando
# ------------------------------------------------------------
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_Legacy/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figure 12/")

# Cargar datos
df = pd.read_csv((Py_Processing_Dir + 'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), low_memory=False, index_col=0)
df = df.reset_index(drop=True)
print('Archivo de Navi cargado')

# Filtrar por bloques de interés y eliminar sujetos específicos
df = df[df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]
df = df[~df['Sujeto'].isin(['P13', 'P05', 'P01'])]

# Contar trials por sujeto y modalidad
conteo_trials = df.groupby(['Sujeto', 'Modalidad'])['Trial_Unique_ID'].nunique().reset_index(name='Trial_Count')
df = conteo_trials

# Invocar CODEX y mapear diagnósticos
c_df = pd.read_excel((Py_Processing_Dir + 'BARANY_CODEX.xlsx'), index_col=0)
Codex_Dict = c_df.to_dict('series')
df['Dx'] = df['Sujeto']
df['Dx'].replace(Codex_Dict['Dg'], inplace=True)
df['Grupo'] = df['Sujeto']
df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)

# Filtrar sujetos y renombrar grupos
df = df[df['Grupo'] != 'P38']
df['Grupo'] = df['Grupo'].replace({
    'MPPP': 'PPPD',
    'Vestibular': 'Vestibular non-PPPD',
    'Voluntario Sano': 'Healthy Volunteer'
})

df.rename(columns={"Grupo": "Group"}, inplace=True)

# Calcular Task Failure
df['Task_Completion'] = (df['Trial_Count'] / 21) * 100
df['Task_Failure'] = 100 - df['Task_Completion']

# Filtrar solo modalidad Realidad Virtual
df = df[df['Modalidad'] == 'Realidad Virtual']

# Definir orden de categorías y colores
categorias_ordenadas = ['PPPD', 'Vestibular non-PPPD', 'Healthy Volunteer']
custom_palette = ["#56B4E9", "#E66100", "#009E73"]  # Azul claro, Naranja, Verde

# Prueba ANOVA
anova_result = stats.f_oneway(
    df[df['Group'] == 'PPPD']['Task_Failure'],
    df[df['Group'] == 'Vestibular non-PPPD']['Task_Failure'],
    df[df['Group'] == 'Healthy Volunteer']['Task_Failure']
)
print(f"ANOVA Result: F = {anova_result.statistic:.3f}, p = {anova_result.pvalue:.4f}")

# Prueba post-hoc Tukey HSD si ANOVA es significativa
tukey_result = None
if anova_result.pvalue < 0.05:
    tukey_result = pairwise_tukeyhsd(df['Task_Failure'], df['Group'])
    print(tukey_result)


# Prueba no paramétrica Kruskal-Wallis
kruskal_result = stats.kruskal(
    df[df['Group'] == 'PPPD']['Task_Failure'],
    df[df['Group'] == 'Vestibular non-PPPD']['Task_Failure'],
    df[df['Group'] == 'Healthy Volunteer']['Task_Failure']
)
print(f"Kruskal-Wallis Result: H = {kruskal_result.statistic:.3f}, p = {kruskal_result.pvalue:.4f}")

# Prueba post-hoc de Dunn con corrección de Bonferroni si Kruskal-Wallis es significativo
dunn_result = None
if kruskal_result.pvalue < 0.05:
    dunn_result = posthoc_dunn(df, val_col='Task_Failure', group_col='Group', p_adjust='bonferroni')
    print(dunn_result)

# Crear gráfico con bordes y barras de error mejoradas
fig, ax = plt.subplots(figsize=(10, 8))

barplot = sns.barplot(
    data=df, x='Group', y='Task_Failure',
    order=categorias_ordenadas, palette=custom_palette,
    linewidth=6, edgecolor="black", errcolor="black", errwidth=6, ax=ax
)

# Fijar posiciones de ticks antes de modificar etiquetas
ax.set_xticks(range(len(categorias_ordenadas)))
ax.set_xticklabels(['PPPD', 'Vestibular\n(non PPPD)', 'Healthy\nVolunteer'], weight='bold', fontsize=14)

# Etiquetas y título
ax.set_ylabel("Percentage of Trials Failed to Complete", fontsize=18, weight='bold', color='black')
ax.set_xlabel(" ", fontsize=18, weight='bold', color='black')
ax.set_title("Virtual Reality Tolerance (% of Task Not Tolerated)", fontsize=18, weight='bold', color='black')
ax.set_ylim(0, 60)

# Agregar estadística post-hoc si corresponde
if dunn_result is not None:
    y_max = 38 #df['Task_Failure'].max()  # Máximo valor de Y
    increment = y_max * 0.18  # Aumentamos el espacio entre líneas

    for i, grupo1 in enumerate(categorias_ordenadas):
        for j, grupo2 in enumerate(categorias_ordenadas):
            if i < j:  # Evitar comparaciones duplicadas
                p_val = dunn_result.loc[grupo1, grupo2]
                if p_val < 0.05:  # Solo si es significativo
                    x1 = categorias_ordenadas.index(grupo1)
                    x2 = categorias_ordenadas.index(grupo2)
                    y = y_max + (i + 1) * increment  # Ajustar altura

                    # Dibujar la línea de significancia
                    ax.plot([x1, x1, x2, x2], [y, y + 2, y + 2, y], color='black', lw=2)

                    # Agregar el asterisco y valores estadísticos
                    ax.text((x1 + x2) / 2, y + 2.5, f"* (p={p_val:.3f})",
                            ha='center', va='bottom', fontsize=14, color='black', fontweight='bold')

# Guardar figura

file_name = f"{Output_Dir}TaskFailure_RV_Only.png"
plt.savefig(file_name)
plt.show()

print(f"-- Gráfico actualizado guardado en {file_name}")
