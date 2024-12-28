#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import socket
from tqdm import tqdm
import time

# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------
print('H-Identifiquemos compu... ')
nombre_host = socket.gethostname()
print(nombre_host)

if nombre_host == 'iMac-de-Hayo.local':
    home = str(Path.home())  # Obtener el directorio raiz en cada computador distinto
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_3/Py_Processing/"

file = Py_Processing_Dir + 'DA_EyeTracker_Synced_NI_2D_withVM.csv'
df_whole = pd.read_csv(file, index_col=0, low_memory=False)
print('Ready')
#%%
df_Fau= df_whole[df_whole['Sujeto'].isin(['P01','P03','P04','P05','P07','P08','P10','P11','P15','P16','P17','P18','P19','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P51','P52','P53'])]

#%%
df_I = df_Fau = df_Fau[df_Fau['MWM_Block'].isin(['HiddenTarget_3'])]
#%%
df=df_I
def factor_fano(data):
    mean = np.mean(data)
    variance = np.var(data)
    return variance / mean if mean != 0 else np.nan

# Primero obtenemos el CSE promedio y el Factor de Fano por cada MWM_Trial
df_trial_summary = df.groupby(['Sujeto', 'MWM_Trial']).agg({
    'CSE': 'mean',
    'x_norm': lambda x: factor_fano(x),
    'y_norm': lambda y: factor_fano(y)
}).reset_index()

# Luego promediamos los valores por Sujeto, considerando todos sus trials
df_subject_summary = df_trial_summary.groupby('Sujeto').agg({
    'CSE': 'mean',  # Promedio de CSE por Sujeto
    'x_norm': 'mean',  # Promedio del Factor de Fano en x_norm por Sujeto
    'y_norm': 'mean'   # Promedio del Factor de Fano en y_norm por Sujeto
}).reset_index()

# Ahora añadimos la columna Grupo, que es única por Sujeto
df_subject_summary = pd.merge(df_subject_summary, df[['Sujeto', 'Grupo']].drop_duplicates(), on='Sujeto')

# Renombramos las columnas para mayor claridad
df_subject_summary.rename(columns={'x_norm': 'Fano_x_norm', 'y_norm': 'Fano_y_norm'}, inplace=True)

#%%
df=df_I
import pandas as pd
import numpy as np
from scipy.signal import convolve2d

# Supongo que ya tienes tu DataFrame 'df'

# Crear una nueva columna con los valores de 'distance' (norma euclidiana de x_norm y y_norm)
df['distance'] = np.sqrt(df['x_norm'] ** 2 + df['y_norm'] ** 2)

# Definir los parámetros para los bins de ocupación
n_place_bins = 128  # Número de bins para la ocupación


# Función para crear la matriz de ocupación y calcular el Factor de Fano
def calcular_factor_fano(df, n_place_bins=128):
    # Filtrar valores fuera de la pantalla (valores entre 0 y 1)
    df_filtered = df[(df['x_norm'] >= 0) & (df['x_norm'] <= 1) & (df['y_norm'] >= 0) & (df['y_norm'] <= 1)]

    # Crear la matriz de ocupación con 'n_place_bins' bins
    heatmap, xedges, yedges = np.histogram2d(df_filtered['x_norm'], df_filtered['y_norm'], bins=n_place_bins)

    # Suavizar la matriz con la convolución usando una ventana de Hanning
    hanning_window = np.hanning(7)
    hanning_kernel = np.outer(hanning_window, hanning_window) / np.sum(hanning_window) ** 2
    smoothed_heatmap = convolve2d(heatmap, hanning_kernel, mode='valid')

    # Calcular el Factor de Fano
    fano_factor = np.mean(smoothed_heatmap) / np.std(smoothed_heatmap) if np.std(smoothed_heatmap) != 0 else np.nan
    return fano_factor


# Agrupar por Sujeto y MWM_Trial y calcular el CSE promedio y el Factor de Fano
df_trial_summary = df.groupby(['Sujeto', 'MWM_Trial']).apply(lambda group: pd.Series({
    'CSE_mean': group['CSE'].mean(),
    'Fano_factor': calcular_factor_fano(group)
})).reset_index()

# Promediar los valores por Sujeto
df_subject_summary = df_trial_summary.groupby('Sujeto').agg({
    'CSE_mean': 'mean',  # Promedio de CSE por Sujeto
    'Fano_factor': 'mean'  # Promedio del Factor de Fano bidimensional por Sujeto
}).reset_index()

# Añadir la columna Grupo, que es única por Sujeto
df_subject_summary = pd.merge(df_subject_summary, df[['Sujeto', 'Grupo']].drop_duplicates(), on='Sujeto')

#%%
from scipy.stats import spearmanr

# Exportar el DataFrame a un archivo Excel
file = Py_Processing_Dir+'FA_Faundez2024.xlsx'

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_subject_summary, x='CSE_mean', y='Fano_factor', hue='Grupo', palette='deep')

# Añadir la línea de tendencia central
sns.regplot(data=df_subject_summary, x='CSE_mean', y='Fano_factor', scatter=False, color='black', ci=None)

# Ajustes de la leyenda y el gráfico
plt.legend(loc='lower left', title='Grupo')  # Ubicar la leyenda en la esquina inferior izquierda
plt.title('Relación entre CSE y Factor de Fano')
plt.xlabel('CSE (Promedio)')
plt.ylabel('Factor de Fano (Promedio)')
plt.grid(False)  # Remover la grilla

# Mostrar el gráfico
plt.show()

# Calcular la correlación de Spearman (no paramétrica)
correlation, p_value = spearmanr(df_subject_summary['CSE_mean'], df_subject_summary['Fano_factor'])

# Imprimir los resultados de la correlación
print(f'Valor de correlación (Spearman): {correlation}')
print(f'P-valor: {p_value}')
