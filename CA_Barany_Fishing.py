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
import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import mnlogit
from scipy.stats import mannwhitneyu
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import scikit_posthocs as sp
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
df = NaviCSE_df.copy()

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

# Identificar sujetos válidos (>=4 True_Trial) en ambas modalidades
valid_trials = conteo_trials[conteo_trials['Trial_Count'] >= 4]
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
sns.set(style="whitegrid")
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
print('Bloque 3 terminado')

#%% ---------------------- Lo mismo que en Bloque 3, pero con estadísticas ---------------
print ('Bloque 4: Estadísticas')
df = Block_Mean_df
bloques_interes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df_filtrado = df[(df['VR_Positive'] == True) & (df['True_Block'].isin(bloques_interes))]

# Orden de los grupos para análisis
orden_grupos = ['MPPP', 'Vestibular', 'Voluntario Sano']

# Análisis entre grupos dentro de cada modalidad
resultados_kruskal = {}
for block in bloques_interes:
    for modalidad in df_filtrado['Modalidad'].unique():
        data = [df_filtrado[(df_filtrado['True_Block'] == block) &
                            (df_filtrado['Modalidad'] == modalidad) &
                            (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna() for grupo in orden_grupos]
        stat, p_value = stats.kruskal(*data)
        resultados_kruskal[(block, modalidad)] = p_value

# Análisis entre modalidades para cada grupo
resultados_wilcoxon = {}
for block in bloques_interes:
    for grupo in orden_grupos:
        data_no_inmersivo = df_filtrado[(df_filtrado['True_Block'] == block) &
                                        (df_filtrado['Modalidad'] == 'No Inmersivo') &
                                        (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()
        data_realidad_virtual = df_filtrado[(df_filtrado['True_Block'] == block) &
                                            (df_filtrado['Modalidad'] == 'Realidad Virtual') &
                                            (df_filtrado['Grupo'] == grupo)]['Block_Mean_Norm_CSE'].dropna()
        if len(data_no_inmersivo) == len(data_realidad_virtual) and len(data_no_inmersivo) > 0:  # Verificar igualdad de longitud y que no estén vacíos
            stat, p_value = stats.wilcoxon(data_no_inmersivo, data_realidad_virtual)
            resultados_wilcoxon[(block, grupo)] = p_value
        else:
            resultados_wilcoxon[(block, grupo)] = 'No se puede calcular, muestras de diferente tamaño o vacías'

# Imprimir resultados
print("Resultados Kruskal-Wallis (Comparación entre grupos dentro de cada modalidad):")
for key, value in resultados_kruskal.items():
    print(f"Block: {key[0]}, Modalidad: {key[1]}, p-value: {value}")

print("\nResultados Wilcoxon (Comparación entre modalidades para cada grupo):")
for key, value in resultados_wilcoxon.items():
    print(f"Block: {key[0]}, Grupo: {key[1]}, p-value: {value}")
#---------------------------------- Ojo con NORmv vs CSE NOrm ----------------
# Análisis entre grupos dentro de cada modalidad
resultados_kruskal = {}
for block in bloques_interes:
    for modalidad in df_filtrado['Modalidad'].unique():
        data = [df_filtrado[(df_filtrado['True_Block'] == block) &
                            (df_filtrado['Modalidad'] == modalidad) &
                            (df_filtrado['Grupo'] == grupo)]['Norm_CSE'].dropna() for grupo in orden_grupos]
        stat, p_value = stats.kruskal(*data)
        resultados_kruskal[(block, modalidad)] = p_value

# Análisis entre modalidades para cada grupo
resultados_wilcoxon = {}
for block in bloques_interes:
    for grupo in orden_grupos:
        data_no_inmersivo = df_filtrado[(df_filtrado['True_Block'] == block) &
                                        (df_filtrado['Modalidad'] == 'No Inmersivo') &
                                        (df_filtrado['Grupo'] == grupo)]['Norm_CSE'].dropna()
        data_realidad_virtual = df_filtrado[(df_filtrado['True_Block'] == block) &
                                            (df_filtrado['Modalidad'] == 'Realidad Virtual') &
                                            (df_filtrado['Grupo'] == grupo)]['Norm_CSE'].dropna()
        if len(data_no_inmersivo) == len(data_realidad_virtual) and len(
                data_no_inmersivo) > 0:  # Verificar igualdad de longitud y que no estén vacíos
            stat, p_value = stats.wilcoxon(data_no_inmersivo, data_realidad_virtual)
            resultados_wilcoxon[(block, grupo)] = p_value
        else:
            resultados_wilcoxon[(block, grupo)] = 'No se puede calcular, muestras de diferente tamaño o vacías'

# Imprimir resultados
print("Resultados Kruskal-Wallis (Comparación entre grupos dentro de cada modalidad):")
for key, value in resultados_kruskal.items():
    print(f"Block: {key[0]}, Modalidad: {key[1]}, p-value: {value}")

print("\nResultados Wilcoxon (Comparación entre modalidades para cada grupo):")
for key, value in resultados_wilcoxon.items():
    print(f"Block: {key[0]}, Grupo: {key[1]}, p-value: {value}")

resultados_mann_whitney = {}
for block in ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']:
    for grupo in ['MPPP', 'Vestibular', 'Voluntario Sano']:
        data_no_inmersivo = df_filtrado[(df_filtrado['True_Block'] == block) &
                                        (df_filtrado['Modalidad'] == 'No Inmersivo') &
                                        (df_filtrado['Grupo'] == grupo)]['Norm_CSE'].dropna()
        data_realidad_virtual = df_filtrado[(df_filtrado['True_Block'] == block) &
                                            (df_filtrado['Modalidad'] == 'Realidad Virtual') &
                                            (df_filtrado['Grupo'] == grupo)]['Norm_CSE'].dropna()
        if not data_no_inmersivo.empty and not data_realidad_virtual.empty:
            stat, p_value = mannwhitneyu(data_no_inmersivo, data_realidad_virtual, alternative='two-sided')
            resultados_mann_whitney[(block, grupo)] = p_value
        else:
            resultados_mann_whitney[(block, grupo)] = 'Datos insuficientes'

# Imprimir resultados
print("Resultados Mann-Whitney U (Comparación entre modalidades para cada grupo):")
for key, value in resultados_mann_whitney.items():
    print(f"Block: {key[0]}, Grupo: {key[1]}, p-value: {value}")

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
sns.histplot(df[df['Modalidad'] == 'No Inmersivo']['Norm_CSE'], color='blue', label='No Inmersivo', kde=True)
sns.histplot(df[df['Modalidad'] == 'Realidad Virtual']['Norm_CSE'], color='red', label='Realidad Virtual', kde=True)
plt.legend()
plt.title('Distribución de CSE después de la normalización')
plt.show()