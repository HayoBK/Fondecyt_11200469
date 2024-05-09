#-----------------------------------------------------
#
#   Una nueva era Comienza, Mayo 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON
#
#-----------------------------------------------------


import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import os
#import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mnlogit
from scipy.stats import mannwhitneyu
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kruskal
#import scikit_posthocs as sp
import socket

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

nombre_host = socket.gethostname()
print(nombre_host)
if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

#%% ----------- NOrmalización
NaviCSE_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
Baldur_df = pd.read_excel((Py_Processing_Dir+'df_Baldur.xlsx'), index_col=0)
df = NaviCSE_df.copy()
df = df.merge(Baldur_df[['Sujeto', 'Dx']], on='Sujeto', how='left')


df = df[df['Sujeto'] != 'P13']
#df = df[df['Sujeto'] != 'P07']
group_stats = df.groupby(['True_Block', 'Modalidad']).agg({
    'CSE': ['mean', 'std']
}).reset_index()

# Renombrar columnas para facilitar el acceso
group_stats.columns = ['True_Block', 'Modalidad', 'Mean_CSE', 'Std_CSE']

# Unir las estadísticas de vuelta al DataFrame original
df = df.merge(group_stats, on=['True_Block', 'Modalidad'])

# Normalizar los valores de CSE
df['Norm_CSE'] = (df['CSE'] - df['Mean_CSE']) / df['Std_CSE']

#%% ----------------- Solo Cargando los datos de Simian y comprobando que esté todo en orden -------------------
print("Bloque 1: Cargado de datos y Revisión de Grupos")
#NaviCSE será la base emergida de SimianMaze con los datos CSE

sujetos_distintos = df['Sujeto'].unique()
sujetos_distintos.sort()
print("Sujetos distintos:", sujetos_distintos)
print('--------------------------------------------------')
conteo_por_grupo = df.groupby('Grupo')['Sujeto'].nunique()
print("Número de sujetos distintos por grupo:")
print(conteo_por_grupo)
print("Bloque 1: Ready")
NaviCSE_df = df.copy()

#%% ----------------- Añadir descriptores de Cuantos pacientes son comparables en realidad virtual.

print('INtento 2 de RV suceptibilidad')
print('Bloque 2 - Ensayo 2: Revisando VR posibility - FUNDAMENTAL para analizar RV')

# Hacer una copia del DataFrame original
df = NaviCSE_df.copy()

# Conteo de los True_Trial distintos por Sujeto, Modalidad, y True_Block
conteo_trials = df.groupby(['Sujeto', 'Modalidad', 'True_Block'])['True_Trial'].nunique().reset_index(name='Trial_Count')
df = df.merge(conteo_trials, on=['Sujeto', 'Modalidad', 'True_Block'], how='left')
df_tc = df.copy()
# Identificar sujetos válidos (>=4 True_Trial) en ambas modalidades
valid_trials = conteo_trials[conteo_trials['Trial_Count'] >= 3]
valid_subjects = valid_trials.pivot_table(index=['Sujeto', 'True_Block'], columns='Modalidad', values='Trial_Count', aggfunc='size', fill_value=0)
valid_subjects['VR_Positive'] = (valid_subjects['No Inmersivo'] > 0) & (valid_subjects['Realidad Virtual'] > 0)
valid_subjects.reset_index(inplace=True)

# Unir esta información al DataFrame original
df = df.merge(valid_subjects[['Sujeto', 'True_Block', 'VR_Positive']], on=['Sujeto', 'True_Block'], how='left')

# Filtrar para obtener solo datos donde VR_Positive es True
df_vr_positive = df[df['VR_Positive'] == True]

# Contar sujetos distintos por True_Block y Grupo, desglosando por cada modalidad

df = df_vr_positive
df = df.merge(conteo_trials, on=['Sujeto', 'Modalidad', 'True_Block'], how='left')

conteo_por_grupo_y_modalidad = df_vr_positive.groupby(['True_Block', 'Grupo', 'Modalidad'])['Sujeto'].nunique().unstack().fillna(0)

# Imprimir los resultados
print("Número de sujetos distintos por grupo, True_Block y Modalidad con VR_Positive:")
print(conteo_por_grupo_y_modalidad)

df_promedio = df.groupby(['Sujeto', 'Modalidad', 'True_Block'])['Norm_CSE'].mean().reset_index(name='Block_Mean_Norm_CSE')
# Unir el promedio calculado de vuelta al DataFrame original
df = pd.merge(df, df_promedio, on=['Sujeto', 'Modalidad', 'True_Block'])
Full_df = df.copy()
# Eliminar duplicados conservando la primera aparición de cada combinación de 'Sujeto', 'Modalidad', y 'True_Block'
Block_Mean_df = df.drop_duplicates(subset=['Sujeto', 'Modalidad', 'True_Block'])


#%%
#---------- Exploremos ahora CSE como parametro al comparar RV y No Inmerviso -----------#
print('Bloque3: Veamos como nos va con el análisis GRAFICOS')
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df = Block_Mean_df
df_filtrado = df[(df['VR_Positive'] == True) & (df['True_Block'].isin(bloques_interes))]

# Configurar el estilo de los gráficos
sns.set(style="white")
group_order = ['MPPP', 'Vestibular', 'Voluntario Sano']

# Crear un boxplot para cada combinación de True_Block y Grupo
plt.figure(figsize=(12, 8))  # Ajusta el tamaño según tus necesidades
for i, block in enumerate(bloques_interes, 1):
    plt.subplot(2, 2, i)  # Ajusta la disposición de los subplots según el número de bloques
    ax= sns.boxplot(x='Grupo', y='Block_Mean_Norm_CSE', hue='Modalidad', data=df_filtrado[df_filtrado['True_Block'] == block], order=group_order)
    plt.title(f'Boxplot de CSE por Grupo y Modalidad en {block}')
    plt.xlabel('Grupo')
    plt.ylabel('Normalized CSE')
    plt.legend(title='Modalidad')
    #ax.set_ylim(0,400)

# Ajustar layout y mostrar el gráfico
plt.tight_layout()
plt.show()


# Create a boxplot for each group combining all blocks
plt.figure(figsize=(10, 6))  # Adjust size as needed
ax = sns.boxplot(x='Grupo', y='Block_Mean_Norm_CSE', hue='Modalidad',
                 data=df_filtrado, order=group_order)
plt.title('Boxplot de CSE por Grupo y Modalidad Combinando Todos los Bloques')
plt.xlabel('Grupo')
plt.ylabel('Normalized CSE')
plt.legend(title='Modalidad')

# Optionally set y-axis limits
# ax.set_ylim(0, 400)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()



# Configurar el estilo de los gráficos
# Configurar el estilo de los gráficos
# Configurar el estilo de los gráficos
sns.set(style="white")
group_order = ['MPPP', 'Vestibular', 'Voluntario Sano']
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']

# Crear un gráfico de puntos para cada combinación de True_Block y Grupo
plt.figure(figsize=(14, 10))  # Ajusta el tamaño según tus necesidades
for i, block in enumerate(bloques_interes, 1):
    plt.subplot(2, 2, i)  # Ajusta la disposición de los subplots según el número de bloques
    ax = sns.stripplot(x='Grupo', y='Block_Mean_Norm_CSE', hue='Modalidad',
                       data=df_filtrado[df_filtrado['True_Block'] == block],
                       order=group_order, dodge=True, jitter=0.25, marker='o', alpha=0.7)

    # Diccionario para almacenar la posición de los sujetos
    pos_dict = {}

    # Guardar posiciones de los puntos para trazar líneas
    for line in range(0, df_filtrado[df_filtrado['True_Block'] == block].shape[0]):
        subject = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Sujeto']
        grupo = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Grupo']
        modalidad = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Modalidad']
        y = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Block_Mean_Norm_CSE']
        x = ax.get_xticks()[group_order.index(grupo)] + (-0.15 if modalidad == 'No Inmersivo' else 0.15)

        if subject not in pos_dict:
            pos_dict[subject] = {}
        pos_dict[subject][modalidad] = (x, y)

        # Anotar sujetos
        color = 'blue' if modalidad == 'No Inmersivo' else 'red'
        ax.text(x, y, subject, horizontalalignment='left', size='x-small', color=color, weight='semibold')

    # Dibujar líneas entre modalidades para el mismo sujeto
    for subject, positions in pos_dict.items():
        if len(positions) == 2:  # Asegurarse de que el sujeto está en ambas modalidades
            x_values, y_values = zip(*positions.values())
            ax.plot(x_values, y_values, color='gray', linestyle='--', linewidth=1, marker='')

    plt.title(f'Puntos de CSE por Grupo y Modalidad en {block}')
    plt.xlabel('Grupo')
    plt.ylabel('Normalized CSE')
    plt.legend(title='Modalidad')

# Ajustar layout y mostrar el gráfico
plt.tight_layout()
plt.show()

ax = sns.stripplot(x='Grupo', y='Block_Mean_Norm_CSE', hue='Modalidad',
                       data=df_filtrado,
                       order=group_order, dodge=True, jitter=0.25, marker='o', alpha=0.7)

    # Diccionario para almacenar la posición de los sujetos
pos_dict = {}

# Guardar posiciones de los puntos para trazar líneas
for line in range(0, df_filtrado[df_filtrado['True_Block'] == block].shape[0]):
    subject = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Sujeto']
    grupo = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Grupo']
    modalidad = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Modalidad']
    y = df_filtrado[(df_filtrado['True_Block'] == block)].iloc[line]['Block_Mean_Norm_CSE']
    x = ax.get_xticks()[group_order.index(grupo)] + (-0.15 if modalidad == 'No Inmersivo' else 0.15)

    if subject not in pos_dict:
        pos_dict[subject] = {}
    pos_dict[subject][modalidad] = (x, y)

    # Anotar sujetos
    color = 'blue' if modalidad == 'No Inmersivo' else 'red'
    ax.text(x, y, subject, horizontalalignment='left', size='x-small', color=color, weight='semibold')

# Dibujar líneas entre modalidades para el mismo sujeto
for subject, positions in pos_dict.items():
    if len(positions) == 2:  # Asegurarse de que el sujeto está en ambas modalidades
        x_values, y_values = zip(*positions.values())
        ax.plot(x_values, y_values, color='gray', linestyle='--', linewidth=1, marker='')

plt.title(f'Puntos de CSE por Grupo y Modalidad en {block}')
plt.xlabel('Grupo')
plt.ylabel('Normalized CSE')
plt.legend(title='Modalidad')

# Ajustar layout y mostrar el gráfico

plt.show()

print('Bloque 3 terminado')

#%% ---------------------- Lo mismo que en Bloque 3, pero con estadísticas ---------------
print ('Bloque 4: Estadísticas')
df = Block_Mean_df
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df_filtrado = df[(df['VR_Positive'] == True) & (df['True_Block'].isin(bloques_interes))]

# Orden de los grupos para análisis
orden_grupos = ['MPPP', 'Vestibular', 'Voluntario Sano']

resultados_mann_whitney = {}
for block in ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']:
    for grupo in ['MPPP', 'Vestibular', 'Voluntario Sano']:
        data_no_inmersivo = df_filtrado[(df_filtrado['True_Block'] == block) &
                                        (df_filtrado['Modalidad'] == 'No Inmersivo') &
                                        (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()
        data_realidad_virtual = df_filtrado[(df_filtrado['True_Block'] == block) &
                                            (df_filtrado['Modalidad'] == 'Realidad Virtual') &
                                            (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()
        if not data_no_inmersivo.empty and not data_realidad_virtual.empty:
            stat, p_value = mannwhitneyu(data_no_inmersivo, data_realidad_virtual, alternative='two-sided')
            resultados_mann_whitney[(block, grupo)] = p_value
        else:
            resultados_mann_whitney[(block, grupo)] = 'Datos insuficientes'

# Imprimir resultados
print("Resultados Mann-Whitney U (Comparación entre modalidades para cada grupo):")
for key, value in resultados_mann_whitney.items():
    print(f"Block: {key[0]}, Grupo: {key[1]}, p-value: {value}")

print ('Finalmente, todos los datos juntos sin diferenciar por bloques')

resultados_mann_whitney = {}
groups = ['MPPP', 'Vestibular', 'Voluntario Sano']
modalities = ['No Inmersivo', 'Realidad Virtual']

# Collecting data across all blocks for each group and modality
for grupo in groups:
    data_no_inmersivo = df_filtrado[(df_filtrado['Modalidad'] == modalities[0]) &
                                    (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()
    data_realidad_virtual = df_filtrado[(df_filtrado['Modalidad'] == modalities[1]) &
                                        (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()

    # Check if both modalities have data before running the test
    if not data_no_inmersivo.empty and not data_realidad_virtual.empty:
        stat, p_value = mannwhitneyu(data_no_inmersivo, data_realidad_virtual, alternative='two-sided')
        resultados_mann_whitney[grupo] = p_value
    else:
        resultados_mann_whitney[grupo] = 'Datos insuficientes'

# Print results
print("Resultados Mann-Whitney U (Comparación entre modalidades para cada grupo combinando todos los bloques):")
for key, value in resultados_mann_whitney.items():
    print(f"Grupo: {key}, p-value: {value}")

#%% Estudio de Normalizaciòn

# EDA antes de la normalización
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Modalidad'] == 'No Inmersivo']['CSE'], color='blue', label='No Inmersivo', kde=True)
sns.histplot(df[df['Modalidad'] == 'Realidad Virtual']['CSE'], color='red', label='Realidad Virtual', kde=True)
plt.legend()
plt.title('Distribución de CSE antes de la normalización')
plt.show()

# EDA después de la normalización
plt.figure(figsize=(12, 6))
sns.histplot(df[df['Modalidad'] == 'No Inmersivo']['Block_Mean_Norm_CSE'], color='blue', label='No Inmersivo', kde=True)
sns.histplot(df[df['Modalidad'] == 'Realidad Virtual']['Block_Mean_Norm_CSE'], color='red', label='Realidad Virtual', kde=True)
plt.legend()
plt.title('Distribución de CSE después de la normalización')
plt.show()

#%% Analisis del DELTA entre modalidades

pivot_df = Block_Mean_df.pivot_table(index=['Sujeto', 'True_Block'], columns='Modalidad', values='Block_Mean_Norm_CSE', aggfunc='first')

# Calcular el Delta como la diferencia entre las modalidades
pivot_df['Delta'] = pivot_df['Realidad Virtual'] - pivot_df['No Inmersivo']
pivot_df.reset_index(inplace=True)

# Fusionar Delta de vuelta a Block_Mean_df
Block_Mean_df = Block_Mean_df.merge(pivot_df[['Sujeto', 'True_Block', 'Delta']], on=['Sujeto', 'True_Block'], how='left')

bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
sns.set(style="whitegrid")  # Establecer el estilo del gráfico

for block in bloques_interes:
    # Filtrar datos por True Block
    df_block = Block_Mean_df[Block_Mean_df['True_Block'] == block]

    # Crear figura
    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(x='Grupo', y='Delta', hue='Grupo', data=df_block, jitter=True, marker='o', alpha=0.7, size=8, order=group_order)

    # Añadir etiquetas de sujeto a cada punto
    subA = 0
    for line in range(df_block.shape[0]):
        sujeto = df_block.iloc[line]['Sujeto']
        delta = df_block.iloc[line]['Delta']
        grupo = df_block.iloc[line]['Grupo']
        x = ax.get_xticks()[list(df_block['Grupo'].unique()).index(grupo)]
        subA=subA+0.005
        ax.text(x+subA, delta, sujeto, horizontalalignment='center', size='x-small', color='black', weight='semibold',
                rotation=0)

    # Configuraciones adicionales del gráfico
    plt.title(f'Delta CSE por Grupo en {block}')
    plt.xlabel('Grupo')
    plt.ylabel('Delta')
    plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para mejor legibilidad
    plt.grid(True)  # Añadir rejilla para facilitar la lectura
    plt.tight_layout()  # Ajustar el layout para evitar solapamientos

    # Mostrar gráfico
    plt.show()

df_block = Block_Mean_df

    # Crear figura
plt.figure(figsize=(10, 6))
ax = sns.stripplot(x='Grupo', y='Delta', hue='Grupo', data=df_block, jitter=True, marker='o', alpha=0.7, size=8, order=group_order)

# Añadir etiquetas de sujeto a cada punto
subA = 0
for line in range(df_block.shape[0]):
    sujeto = df_block.iloc[line]['Sujeto']
    delta = df_block.iloc[line]['Delta']
    grupo = df_block.iloc[line]['Grupo']
    x = ax.get_xticks()[list(df_block['Grupo'].unique()).index(grupo)]
    subA=subA+0.005
    ax.text(x+subA, delta, sujeto, horizontalalignment='center', size='x-small', color='black', weight='semibold',
            rotation=0)

# Configuraciones adicionales del gráfico
plt.title(f'Delta CSE por Grupo en {block}')
plt.xlabel('Grupo')
plt.ylabel('Delta')
plt.xticks(rotation=45)  # Rotar las etiquetas del eje X para mejor legibilidad
plt.grid(True)  # Añadir rejilla para facilitar la lectura
plt.tight_layout()  # Ajustar el layout para evitar solapamientos

# Mostrar gráfico
plt.show()

#%% ---------------- Veamos Migraña Vestibular
df=Block_Mean_df
df['Dx_MPPP'] = (df['Grupo'] == 'MPPP').astype(int)

# Crear Dx_MV basado en la columna Dx si contiene 'MV'
df['Dx_MV'] = df['Dx'].str.contains('MV', na=False).astype(int)


conditions = [
    (df['Dx_MPPP'] == 1) & (df['Dx_MV'] == 1),
    (df['Dx_MPPP'] == 1) & (df['Dx_MV'] == 0),
    (df['Dx_MPPP'] == 0) & (df['Dx_MV'] == 1),
    (df['Dx_MPPP'] == 0) & (df['Dx_MV'] == 0)
]

# Nombres de las categorías
categories = ['Both Dx_MPPP and Dx_MV', 'Only Dx_MPPP', 'Only Dx_MV', 'Neither']

# Crear una nueva columna 'Category'
df['Category'] = pd.np.select(conditions, categories)

plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Category', y='Block_Mean_Norm_CSE', data=df)
plt.title('Comparación de Block_Mean_Norm_CSE por Categoría de Diagnóstico')
plt.xlabel('Categoría de Diagnóstico')
plt.ylabel('Block Mean Normalized CSE')
plt.xticks(rotation=45)  # Rotar las etiquetas para mejor visualización
plt.show()

data_groups = [df[df['Category'] == cat]['Block_Mean_Norm_CSE'] for cat in categories]

# Realizar la prueba de Kruskal-Wallis
stat, p_value = kruskal(*data_groups)

print(f"Kruskal-Wallis H-test result: Stat={stat}, P-value={p_value}")

plt.figure(figsize=(12, 8))
ax = sns.boxplot(x='Category', y='Delta', data=df)
plt.title('Comparación de Delta por Categoría de Diagnóstico')
plt.xlabel('Categoría de Diagnóstico')
plt.ylabel('Delta')
plt.xticks(rotation=45)  # Rotar las etiquetas para mejor visualización
plt.show()

data_groups = [df[df['Category'] == cat]['Delta'] for cat in categories]

# Realizar la prueba de Kruskal-Wallis
stat, p_value = kruskal(*data_groups)
print('Comparación de Delta por Categoría de Diagnóstico')
print(f"Kruskal-Wallis H-test result: Stat={stat}, P-value={p_value}")

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x='Dx_MV', y='Delta', data=df)
plt.title('Comparación de Delta por Presencia de Dx_MV')
plt.xlabel('Categoría de Dx_MV')
plt.ylabel('Delta')
plt.show()

group_dx_mv = df[df['Dx_MV'] == 1]['Delta']
group_no_dx_mv = df[df['Dx_MV'] == 0]['Delta']

# Realizar la prueba de Mann-Whitney U
u_stat, p_value = mannwhitneyu(group_dx_mv, group_no_dx_mv, alternative='two-sided')

print(f"Mann-Whitney U test result: U-statistic={u_stat}, P-value={p_value}")


print('Bloque3: Veamos como nos va con el análisis GRAFICOS')
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df = Block_Mean_df
df_filtrado = df[(df['VR_Positive'] == True) & (df['True_Block'].isin(bloques_interes))]

# Configurar el estilo de los gráficos
sns.set(style="white")
group_order = ['MPPP', 'Vestibular', 'Voluntario Sano']

# Crear un boxplot para cada combinación de True_Block y Grupo
plt.figure(figsize=(12, 8))  # Ajusta el tamaño según tus necesidades
for i, block in enumerate(bloques_interes, 1):
    plt.subplot(2, 2, i)  # Ajusta la disposición de los subplots según el número de bloques
    ax= sns.boxplot(x='Grupo', y='Delta', hue='Modalidad', data=df_filtrado[df_filtrado['True_Block'] == block], order=group_order)
    plt.title(f'Boxplot de Delta-CSE por Grupo y Modalidad en {block}')
    plt.xlabel('Grupo')
    plt.ylabel('Normalized Delta-CSE')
    plt.legend(title='Modalidad')
    #ax.set_ylim(0,400)

# Ajustar layout y mostrar el gráfico
plt.tight_layout()
plt.show()

bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
group_order = ['MPPP', 'Vestibular', 'Voluntario Sano']

# Resultados por bloque
results = {}
for block in bloques_interes + ['All']:  # Incluir un análisis para todos los bloques juntos
    if block == 'All':
        data_block = df_filtrado  # Considerar todos los datos si es el análisis conjunto
    else:
        data_block = df_filtrado[df_filtrado['True_Block'] == block]

    # Recolectar datos por grupo
    data_by_group = [data_block[data_block['Grupo'] == group]['Delta'] for group in group_order]

    # Kruskal-Wallis test
    stat, p_value = kruskal(*data_by_group)
    results[block] = {'statistic': stat, 'p_value': p_value}

# Mostrar los resultados
for block, res in results.items():
    print(f"Kruskal-Wallis test for {block}: Statistic={res['statistic']}, P-value={res['p_value']}")


df = df_tc
df_first_entry = df.drop_duplicates(subset=['Sujeto', 'Modalidad', 'True_Block'], keep='first')
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']

# Crear una figura para contener los subgráficos
fig, axes = plt.subplots(nrows=len(bloques_interes), ncols=1, figsize=(10, 6 * len(bloques_interes)))

# Si solo hay un bloque, convertir axes en un array
if len(bloques_interes) == 1:
    axes = [axes]

# Crear un gráfico para cada True_Block
for ax, block in zip(axes, bloques_interes):
    sns.barplot(x='Grupo', y='Trial_Count', hue='Modalidad', data=df_first_entry[df_first_entry['True_Block'] == block], ax=ax)
    ax.set_title(f'Trial Count por Grupo y Modalidad en {block}')
    ax.set_xlabel('Grupo')
    ax.set_ylabel('Trial Count')
    ax.legend(title='Modalidad')

# Ajustar el layout
plt.tight_layout()
plt.show()

grupos = df_first_entry['Grupo'].unique()
bloques = df_first_entry['True_Block'].unique()

# Crear una figura para cada combinación de grupo y bloque
for grupo in grupos:
    for bloque in bloques:
        # Filtrar los datos por grupo y bloque
        data_group_block = df_first_entry[(df_first_entry['Grupo'] == grupo) & (df_first_entry['True_Block'] == bloque)]

        # Verificar si hay datos para plotear
        if data_group_block.empty:
            continue

        # Crear la figura y el eje
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Sujeto', y='Trial_Count', hue='Modalidad', data=data_group_block)

        # Configurar título y etiquetas
        ax.set_title(f'Trial Count por Sujeto en el Grupo {grupo}, Bloque {bloque}')
        ax.set_xlabel('Sujeto')
        ax.set_ylabel('Trial Count')
        ax.legend(title='Modalidad')

        # Mejorar la visualización de las etiquetas del eje X
        plt.xticks(rotation=45)

        # Mostrar el gráfico
        plt.show()