#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 4 dias.
#   PUPIL LABS INFO
#-----------------------------------------------------


#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import socket
from tqdm import tqdm
import time

from DA_For_Barany2024 import categorias

# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------
print('H-Identifiquemos compu... ')
nombre_host = socket.gethostname()
print(nombre_host)

if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

if nombre_host == 'MSI':
    home="D:/Titan-OneDrive"
    home_path = Path("D:/Titan-OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/Barany2024/"

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"

print('Compu identificado.')

file = Py_Processing_Dir+'DA_EyeTracker_Synced_NI_2D_withVM.csv'
df_whole = pd.read_csv(file, index_col=0, low_memory=False)
df_whole = df_whole.copy()
df_whole = df_whole[~df_whole['Sujeto'].isin(['P05','P14','P16'])]

df_whole['4X-Code']=df_whole['Sujeto']+df_whole['Categoria'] # Añadimos un codigo de Sujeto unico nuevo para evitar sujetos repetidos por la asignación de la
                                                                # Categoria MV
Esto_si = False
if Esto_si:
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    #Vamos a añadir Scanned Path de Pedro.-----------------------------------------------------------------------------------------------------
    df_whole['delta_x'] = df_whole['x_norm'].diff()
    df_whole['delta_y'] = df_whole['y_norm'].diff()

    # Calcula la distancia euclidiana
    df_whole['distancia'] = np.sqrt(df_whole['delta_x']**2 + df_whole['delta_y']**2)

    # Opcionalmente, puedes eliminar las columnas temporales 'delta_x' y 'delta_y'
    df_whole = df_whole.drop(columns=['delta_x', 'delta_y'])

    # Para la primera fila, donde no hay fila previa, puedes reemplazar NaN con 0 o algún otro valor si lo prefieres
    df_whole['distancia'].fillna(0, inplace=True)

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    # Vamos a Generar duraciones para cada OW_Trial, para asegurar un peso equivalente entre cada OW_a traves de sujetos.......-----------------------------------
    #df_whole['timestamp'] = pd.to_datetime(df_whole['timestamp'])

    # Agrupamos por '4X-Code' y 'OW_Trial' y calculamos la duración de cada trial
    duraciones = df_whole.groupby(['4X-Code', 'OW_Trial'])['timestamp'].agg(Inicio='min', Fin='max')
    duraciones['Duracion'] = duraciones['Fin'] - duraciones['Inicio']

    # Unimos la información de duración de vuelta al DataFrame original
    df_whole = df_whole.merge(duraciones['Duracion'], on=['4X-Code', 'OW_Trial'], how='left')

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    #Ahora haremos distancias sumadas, y luego lo haremos por unidad de tiempo para normalizar por trials de distinta duracion...--------------------------------

    # Asegúrate de que ya has calculado las distancias y las duraciones como hemos discutido anteriormente

    # Paso 1: Sumar las distancias por grupo (4X-Code, OW_Trial)
    distancia_sumada = df_whole.groupby(['4X-Code', 'OW_Trial'])['distancia'].sum().reset_index()

    # Renombramos la columna para mayor claridad
    distancia_sumada.rename(columns={'distancia': 'distancia_sumada'}, inplace=True)

    # Unimos la columna de distancia sumada al DataFrame original
    df_whole = df_whole.merge(distancia_sumada, on=['4X-Code', 'OW_Trial'], how='left')

    # Paso 2: Calcular la "distancia por unidad de tiempo"
    df_whole['Scanned Path / time'] = df_whole['distancia_sumada'] / df_whole['Duracion']

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # Falto reducir por MWM_Block
    #------------------------------------------------------------------------------------------------------------------------------------------------------



    promedios = df_whole.groupby(['4X-Code', 'MWM_Block'])['Scanned Path / time'].mean().reset_index()

    # Renombramos la columna para mayor claridad
    promedios.rename(columns={'Scanned Path / time': 'Scanned_Path_per_time_per_Block'}, inplace=True)

    # Ahora puedes unir estos promedios al DataFrame original o trabajar solo con el DataFrame de promedios
    df_whole = df_whole.merge(promedios, on=['4X-Code', 'MWM_Block'], how='left')


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Vamonos a graficar
#------------------------------------------------------------------------------------------------------------------------------------------------------


Bloques_de_Interes = []
Bloques_de_Interes.append(['Todo',[]])
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

# Seleccionar las columnas relevantes para identificar duplicados
columnas_para_unicas = ['4X-Code', 'MWM_Block']

# Eliminar filas duplicadas basadas en estas columnas
df_reducido = df_whole.drop_duplicates(subset=columnas_para_unicas, keep='first').reset_index(drop=True)

columnas_para_unicas = ['4X-Code', 'OW_Trial']

# Eliminar filas duplicadas basadas en estas columnas
df_read = df_whole.drop_duplicates(subset=columnas_para_unicas, keep='first').reset_index(drop=True)

df_reducido['4X-Code'] = df_reducido['4X-Code'].str[:6]
for Bl in Bloques_de_Interes:
    if Bl[1]:
        data=df_reducido[df_reducido['MWM_Block'].isin(Bl[1])]
    else:
        data=df_reducido
    print(f"Generando Grafico para {Bl[0]}")
    ax = sns.boxplot(data, x='Categoria', y='Scanned_Path_per_time_per_Block', linewidth=6, order=categorias_ordenadas, hue='Categoria', legend=False)
    sns.stripplot(data=data, x='Categoria', y='Scanned_Path_per_time_per_Block', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
    offsets = ax.collections[-1].get_offsets()
    for i, (x, y) in enumerate(offsets):
        ax.annotate(data.iloc[i]['4X-Code'], (x, y),
                    ha='center', va='center', fontsize=8, color='black')
    ax.set_ylabel("Gaze Scanned Path / Time ", weight='bold')
    ax.set_xlabel("Category", weight='bold')
    ax.set_xticks(range(len(categorias_ordenadas)))
    ax.set_xticklabels(categorias_ordenadas)

    ax.set(ylim=(0, 1000))
    #ax.set_title(Title, weight='bold')
    # Determine the y position for the line and annotation
    file_name = f"{Output_Dir}New_{Bl[0]}_Gaze_Scanned_Path.png"
    plt.savefig(file_name)
    plt.clf()
    print(f"--Completo Grafico para {Bl[0]}")

df_whole = df_whole[df_whole['distancia'] >= 0.01]
sujetos_unicos = df_whole['4X-Code'].unique()
print(sujetos_unicos)

for Bl in Bloques_de_Interes:

    df= df_whole[df_whole['MWM_Block'].isin(Bl[1])]

    for categoria in categorias_ordenadas:
        print('Iniciando Procesamiento de ', Bl[0], categoria)

        for S in sujetos_unicos:
            data = df[df['Categoria'] == categoria]
            data=data[data['4X-Code']==S]
            data = data[['x_norm', 'y_norm']]
            #print(data.shape[0])
            if data.shape[0] > 1000:  # Downsample if necessary
                data = data.sample(1000, random_state=42)
            #print(data.shape[0])
            #print('sub-df generada')
            if data.shape[0] > 0:
                # Recorremos las categorías desde la lista predefinida
                print(f'Generando gráfico para la categoría: {categoria} en {Bl[0]}')
                sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=100, thresh=0, fill=True,
                            cbar=True)
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                # Añadimos líneas de guía
                for val in [0.25, 0.75]:
                    plt.axvline(val, color='gray', linestyle='--', lw=1)
                    plt.axhline(val, color='gray', linestyle='--', lw=1)
                plt.axvline(0.5, color='gray', linestyle='--', lw=2)
                plt.axhline(0.5, color='gray', linestyle='--', lw=2)

                # Añadimos el título
                plt.title(f'Gaze Distribution for {categoria} in {Bl[0]}')

                # Guardamos el gráfico
                file_name = f"{Output_Dir}2Dper4X-Code/{categoria} {S} {Bl[0]}_Gaze_Distribution.png"
                plt.savefig(file_name)
                plt.clf()

        inicio_bloque = time.time()
        data = df[df['Categoria']==categoria]
        data=data[['x_norm', 'y_norm']]
        #print(data.shape[0])
        if data.shape[0] > 1000:  # Downsample if necessary
            data = data.sample(1000, random_state=42)
        #print(data.shape[0])
        #print('sub-df generada')
    # Recorremos las categorías desde la lista predefinida
        print(f'Generando gráfico MAYOR para la categoría: {categoria} en {Bl[0]}')
        sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=100, thresh=0, fill=True, cbar=True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

# Añadimos líneas de guía
        for val in [0.25, 0.75]:
            plt.axvline(val, color='gray', linestyle='--', lw=1)
            plt.axhline(val, color='gray', linestyle='--', lw=1)
        plt.axvline(0.5, color='gray', linestyle='--', lw=2)
        plt.axhline(0.5, color='gray', linestyle='--', lw=2)

# Añadimos el título
        plt.title(f'Gaze Distribution for {categoria} in {Bl[0]}')

# Guardamos el gráfico
        file_name = f"{Output_Dir}2DperCat/{Bl[0]}_Gaze_Distribution_{categoria}_.png"
        plt.savefig(file_name)
        plt.clf()
        duracion = time.time() - inicio_bloque
        print(f'Terminando Procesamiento de {categoria} {Bl[0]} en {duracion:.2f} segundos')

print('And Dream of the Endless asked: What do you dream of then, when you dream of sex?')
#%%
print('Final real del archivo')
