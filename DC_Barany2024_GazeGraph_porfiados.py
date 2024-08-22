#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 4 dias.
#   Version TRAVELLING!
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
    # Directorios version 2024 Agosto 22
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Outputs/Barany2024/"

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"

print('Compu identificado.')
#----------------------------------------------------------------------------------------------------------
#           Grafico 1
#----------------------------------------------------------------------------------------------------------

file = Py_Processing_Dir+'DA_Gaze_2D_reducido_HT_1.csv'
df = pd.read_csv(file, index_col=0, low_memory=False)
data=df
categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

fig, ax = plt.subplots(figsize=(10, 8))
color_mapping = {
    'PPPD': "#ADD8E6",
    'Vestibular Migraine': "#DDA0DD",
    'Vestibular (non PPPD)': "#FFA07A" ,
    'Healthy Volunteer': "#98FB98"
}
#FFA07A: Light Salmon
#DDA0DD: Plum
#ADD8E6: Light Blue
#98FB98: Pale Green

custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
ax = sns.boxplot(data, x='Categoria', y='Scanned_Path_per_time_per_Block', linewidth=6, order=categorias_ordenadas, hue='Categoria', hue_order=categorias_ordenadas, legend=False, palette=custom_palette)
sns.stripplot(data=data, x='Categoria', y='Scanned_Path_per_time_per_Block', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
offsets = ax.collections[-1].get_offsets()

ax.set_ylabel("Gaze Scanned Path / Time ", fontsize=18, weight='bold', color='darkblue')
ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')

ax.set_xticks(range(len(categorias_ordenadas)))
ax.set_xticklabels(categorias_ordenadas)

ax.set(ylim=(0, 50))
ax.set_title("Scanned_Path over 2D-Screen (Non-Immersive) ", fontsize=18, weight='bold', color='darkblue')
# Determine the y position for the line and annotation
file_name = f"{Output_Dir}Porfiados/Gaze_Scanned_Path_2D.png"
plt.savefig(file_name)
plt.clf()

#----------------------------------------------------------------------------------------------------------
#           Grafico 2
#----------------------------------------------------------------------------------------------------------

file = Py_Processing_Dir+'DA_Gaze_RV_reducido_HT_2.csv'
df = pd.read_csv(file, index_col=0, low_memory=False)
data=df
categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

fig, ax = plt.subplots(figsize=(10, 8))
color_mapping = {
    'PPPD': "#ADD8E6",
    'Vestibular Migraine': "#DDA0DD",
    'Vestibular (non PPPD)': "#FFA07A" ,
    'Healthy Volunteer': "#98FB98"
}
#FFA07A: Light Salmon
#DDA0DD: Plum
#ADD8E6: Light Blue
#98FB98: Pale Green
data['Scanned_Path_per_time_per_Block'] = data['Scanned_Path_per_time_per_Block'] / 10


custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
ax = sns.boxplot(data, x='Categoria', y='Scanned_Path_per_time_per_Block', linewidth=6, order=categorias_ordenadas, hue='Categoria', hue_order=categorias_ordenadas, legend=False, palette=custom_palette)
sns.stripplot(data=data, x='Categoria', y='Scanned_Path_per_time_per_Block', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
offsets = ax.collections[-1].get_offsets()

ax.set_ylabel("Gaze Scanned Path / Time ", fontsize=18, weight='bold', color='darkblue')
ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')

ax.set_xticks(range(len(categorias_ordenadas)))
ax.set_xticklabels(categorias_ordenadas)

ax.set(ylim=(0, 50))
ax.set_title("Scanned_Path over Virtual Reality -Normalized Projection", fontsize=18, weight='bold', color='darkblue')
# Determine the y position for the line and annotation
file_name = f"{Output_Dir}Porfiados/Gaze_Scanned_Path_VR.png"
plt.savefig(file_name)
plt.clf()

print('Protocolo Porfiados Listos. ')