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
df_whole = pd.read_csv(file, index_col=0)


categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

for Bl in Bloques_de_Interes:
    print('Iniciando Procesamiento de ',Bl[0])
    inicio_bloque = time.time()

    df= df_whole[df_whole['True_Block'].isin(Bl[1])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Recorremos las categorías desde la lista predefinida
    for ax, categoria in zip(axes.flatten(), categorias_ordenadas):
        # Filtramos el DataFrame por la categoría actual
        data = df[df['Categoria'] == categoria]

        # Generamos el mapa de calor
        sns.kdeplot(
            x=data['x_norm'], y=data['y_norm'],
            fill=True, cmap='coolwarm',
            ax=ax, levels=20, thresh=0
        )

        # Ajustamos los límites del eje
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Añadimos líneas de guía
        for val in [0.25, 0.75]:
            ax.axvline(val, color='gray', linestyle='--', lw=1)
            ax.axhline(val, color='gray', linestyle='--', lw=1)
        ax.axvline(0.5, color='gray', linestyle='--', lw=2)
        ax.axhline(0.5, color='gray', linestyle='--', lw=2)

        # Añadimos el título de cada subplot
        ax.set_title(f'Categoria: {categoria}')

    # Ajustamos los espacios entre subplots
    plt.tight_layout()
    Title= "Gaze Distribution on Screen " + Bl[0]
    plt.savefig(Output_Dir + '02a - Gaze2D/'+ Title + '.png')
    plt.clf()
    #plt.show()
    duracion = time.time() - inicio_bloque
    print(f'Terminando Procesamiento de {Bl[0]} en {duracion:.2f} segundos')
