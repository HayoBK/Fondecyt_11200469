#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 10 dias.
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


# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------
print('H-Iniciando segmento... ')
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

file = Py_Processing_Dir+'DA_EyeTracker_Synced_NI_2D_withVM.csv'
df_whole = pd.read_csv(file, index_col=0, low_memory=False)
df_whole = df_whole[df_whole['on_surf']==True]
#df_whole = df_whole.dropna(subset=['x_norm', 'y_norm'])



categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

for Bl in Bloques_de_Interes:


    df= df_whole[df_whole['MWM_Block'].isin(Bl[1])]
    for categoria in categorias_ordenadas:
        print('Iniciando Procesamiento de ', Bl[0], categoria)
        inicio_bloque = time.time()
        data = df[df['Categoria']==categoria]
        data=data[['x_norm', 'y_norm']]
        print(data.shape[0])
        if data.shape[0] > 10000:  # Downsample if necessary
            data = data.sample(10000, random_state=42)
        print(data.shape[0])
        print('sub-df generada')
    # Recorremos las categorías desde la lista predefinida
        print(f'Generando gráfico para la categoría: {categoria} en {Bl[0]}')
        sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=20, thresh=0, fill=True, cbar=True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)

# Añadimos líneas de guía
        #for val in [0.25, 0.75]:
        #    plt.axvline(val, color='gray', linestyle='--', lw=1)
        #    plt.axhline(val, color='gray', linestyle='--', lw=1)
        #plt.axvline(0.5, color='gray', linestyle='--', lw=2)
        #plt.axhline(0.5, color='gray', linestyle='--', lw=2)

# Añadimos el título
        plt.title(f'Gaze Distribution for {categoria} in {Bl[0]}')

# Guardamos el gráfico
        file_name = f"{Output_Dir}02a - Gaze2D/Gaze_Distribution_{categoria}_{Bl[0]}.png"
        plt.savefig(file_name)
        plt.clf()
        duracion = time.time() - inicio_bloque
        print(f'Terminando Procesamiento de {categoria} {Bl[0]} en {duracion:.2f} segundos')



#%%
print('God in his heaven, all is right on Earth')