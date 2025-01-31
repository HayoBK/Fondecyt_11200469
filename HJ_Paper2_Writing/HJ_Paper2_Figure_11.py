#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 1 dias.
#   Version En AEROPUERTOS
#   PUPIL LABS INFO
#-----------------------------------------------------
# Me gustó: Total_Angular Movement Summed

#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import socket
from tqdm import tqdm
import time
from pathlib import Path
import HA_ModuloArchivos as H_Mod
import scipy.stats as stats
from scikit_posthocs import posthoc_dunn

# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_Legacy/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figure 11/")

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage5.csv'
df = pd.read_csv(file, index_col=0, low_memory=False)

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])
df = df[df["Categoria"] != "Vestibular Migraine"]

categorias_ordenadas = ['PPPD', 'Vestibular (non PPPD)', 'Healthy Volunteer']
Ejes = ['vRoll_normalizada_por_Bloque','vYaw_normalizada_por_Bloque','vPitch_normalizada_por_Bloque','AngMagnitud_normalizada_por_Bloque']



color_mapping = {
    'PPPD': "#56B4E9",
    'Vestibular (non PPPD)': "#E66100" ,
    'Healthy Volunteer': "#009E73"
}

for Bl in Bloques_de_Interes:
    if Bl[1]:
        data=df[df['MWM_Block'].isin(Bl[1])]
    else:
        data=df
    print(f"Generando Grafico para {Bl[0]}")
    for idx, Ex in enumerate(Ejes):

        fig, ax = plt.subplots(figsize=(10, 8))
        custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
        ax = sns.boxplot(data=data, x='Categoria', y=Ex, linewidth=6, order=categorias_ordenadas, palette=custom_palette)
        #offsets = ax.collections[-1].get_offsets()
        #for i, (x, y) in enumerate(offsets):
        #    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
        #                ha='center', va='center', fontsize=8, color='black')
        ax.set_ylabel(Ex, fontsize=18, weight='bold', color='black')
        ax.set_xlabel("Category", fontsize=18, weight='bold', color='black')
        #ax.set_xticks(range(len(categorias_ordenadas)))
        #ax.set_xticklabels(categorias_ordenadas)
        #ax.get_legend().remove()
        ax.set(ylim=(0, 50))
        if idx == 3:
            ax.set(ylim=(0, 100))
        Title = f"Head Kinematics for {Bl[0]}"
        ax.set_title(Title, fontsize=18, weight='bold', color='black')
        # Determine the y position for the line and annotation
        file_name = f"{Output_Dir}{Ex}_{Bl[0]}_Angular Movement Summed.png"
        plt.savefig(file_name)
        plt.clf()
        print(f"--Completo Grafico para {Ex} & {Bl[0]}")

df['4X-Code'] = df['Sujeto'] + df['Categoria']
data = df.groupby(['4X-Code', 'Categoria']).agg({'AngMagnitud_normalizada_por_Bloque': ['mean']}).reset_index()
data.columns = ['4X-Code', 'Categoria', 'Ang']
fig, ax = plt.subplots(figsize=(10, 8))


# PRUEBA DE KRUSKAL-WALLIS
kruskal_result = stats.kruskal(
    data[data['Categoria'] == 'PPPD']['Ang'],
    data[data['Categoria'] == 'Vestibular (non PPPD)']['Ang'],
    data[data['Categoria'] == 'Healthy Volunteer']['Ang']
)

print(f"Kruskal-Wallis Result: H = {kruskal_result.statistic:.3f}, p = {kruskal_result.pvalue:.4f}")

# PRUEBA POST-HOC DE DUNN CON CORRECCIÓN DE BONFERRONI SI KRUSKAL ES SIGNIFICATIVA
dunn_result = None
if kruskal_result.pvalue < 0.05:
    dunn_result = posthoc_dunn(data, val_col='Ang', group_col='Categoria', p_adjust='bonferroni')
    print(dunn_result)


custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
ax = sns.boxplot(data=data, x='Categoria', y='Ang', linewidth=6, order=categorias_ordenadas, palette=custom_palette)
sns.stripplot(data=data, x='Categoria', y='Ang', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
offsets = ax.collections[-1].get_offsets()
#for i, (x, y) in enumerate(offsets):
#    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
#                ha='center', va='center', fontsize=8, color='black')
ax.set_ylabel(" Summed angular motion in all axis (degrees) ", fontsize=18, weight='bold', color='black')
ax.tick_params(axis='x', labelsize=18, rotation=0)

ax.set_xlabel("Group", fontsize=18, weight='bold', color='black')
#ax.set_xticks(range(len(categorias_ordenadas)))
#ax.set_xticklabels(categorias_ordenadas)
#ax.get_legend().remove()
ax.set(ylim=(0, 50))
if idx == 3:
    ax.set(ylim=(0, 100))


# AGREGAR SIGNIFICANCIA ESTADÍSTICA
if dunn_result is not None:
    y_max = data['Ang'].max()
    increment = y_max * 0.08

    for i, grupo1 in enumerate(categorias_ordenadas):
        for j, grupo2 in enumerate(categorias_ordenadas):
            if i < j:  # Evitar comparaciones duplicadas
                p_val = dunn_result.loc[grupo1, grupo2]
                if p_val < 0.05:  # Solo si es significativo
                    x1 = categorias_ordenadas.index(grupo1)
                    x2 = categorias_ordenadas.index(grupo2)
                    y = y_max + (i + 1) * increment
                    ax.plot([x1, x1, x2, x2], [y, y + 2, y + 2, y], color='black', lw=2)
                    ax.text((x1 + x2) / 2, y + 3, f"* (p={p_val:.3f})", ha='center', va='bottom', fontsize=14, color='black', fontweight='bold')



Title = f"Head Kinematics: Total head movement (angular)"
ax.set_title(Title, fontsize=18, weight='bold', color='black')
# Determine the y position for the line and annotation
file_name = f"{Output_Dir}Total_Angular Movement Summed.png"
plt.savefig(file_name)
plt.clf()
print(f"--Completo Grafico para")



print('Final de script completo')