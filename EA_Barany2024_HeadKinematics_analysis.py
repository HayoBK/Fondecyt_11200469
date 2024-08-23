#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 1 dias.
#   Version En AEROPUERTOS
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
from pathlib import Path

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
    # Directorios version 2024 Agosto 22
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Outputs/Barany2024/"

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"

if nombre_host == 'Hayos-MacBook-Pro.local':
    home = str(Path.home())
    Py_Processing_Dir = home + "/Py_Adventure/PyPro_traveling_3/Py_Processing/"
    Output_Dir = home + "/Py_Adventure/PyPro_traveling_3/Outputs/Barany2024_Air/"
print('Compu identificado.')

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage5.csv'
df = pd.read_csv(file, index_col=0, low_memory=False)

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']
Ejes = ['vRoll_normalizada_por_Bloque','vYaw_normalizada_por_Bloque','vPitch_normalizada_por_Bloque','AngMagnitud_normalizada_por_Bloque']

color_mapping = {
    'PPPD': "#ADD8E6",
    'Vestibular Migraine': "#DDA0DD",
    'Vestibular (non PPPD)': "#FFA07A" ,
    'Healthy Volunteer': "#98FB98"
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
        sns.stripplot(data=data, x='Categoria', y=Ex, jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
        offsets = ax.collections[-1].get_offsets()
        #for i, (x, y) in enumerate(offsets):
        #    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
        #                ha='center', va='center', fontsize=8, color='black')
        ax.set_ylabel(Ex, fontsize=18, weight='bold', color='darkblue')
        ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')
        #ax.set_xticks(range(len(categorias_ordenadas)))
        #ax.set_xticklabels(categorias_ordenadas)
        #ax.get_legend().remove()
        ax.set(ylim=(0, 50))
        if idx == 3:
            ax.set(ylim=(0, 100))
        Title = f"Head Kinematics for {Bl[0]}"
        ax.set_title(Title, fontsize=18, weight='bold', color='darkblue')
        # Determine the y position for the line and annotation
        file_name = f"{Output_Dir}HeadKinematics/{Ex}_{Bl[0]}_Angular Movement Summed.png"
        plt.savefig(file_name)
        plt.clf()
        print(f"--Completo Grafico para {Ex} & {Bl[0]}")
#%%
print('Final de script completo')