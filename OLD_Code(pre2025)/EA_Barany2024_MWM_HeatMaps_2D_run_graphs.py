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

df = pd.read_csv((Py_Processing_Dir+'EA_MWM_Position_withMV_2D.csv'), low_memory=False, index_col=0)
df= df.reset_index(drop=True)
print('Archivo de Navi cargado')
def remove_extreme_timestamps(group):
    # Sort by timestamp just in case it's not sorted
    group = group.sort_values(by='P_timeMilliseconds')

    # Calculate the indices for the 20% and 80% thresholds
    lower_index = int(len(group) * 0.15)
    upper_index = int(len(group) * 0.9999)

    # Keep only the middle 60% of the data
    return group.iloc[lower_index:upper_index]
def MapaDeCalor(dat, Mod, Bloc, Grupo, Titulo):
    colores = sns.color_palette('coolwarm',n_colors=100)
    dat = dat.loc[(dat['Categoria'] == Grupo) & (dat['Modalidad'] == Mod) & (dat['True_Block'] == Bloc)]
    if dat.shape[0] > 0:
        Title = str(Titulo) + str(Grupo)
        dat.reset_index()
        dat = remove_extreme_timestamps(dat)
        sns.set_style("ticks")
        sns.set(style='ticks',rc={"axes.facecolor":colores[0]},font_scale=1.5)
        ax = sns.kdeplot(data=dat, x='P_position_x', y='P_position_y', cmap='coolwarm', n_levels=75, thresh=0, fill=True, cbar=False)

        sns.set_context(font_scale=3)
        ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)
        ax.tick_params(labelsize=13)
        ax.set_title(Title, fontsize=16, weight='bold', color='darkblue')
        circle = plt.Circle((0, 0), 0.5, color='w',linewidth= 2, fill=False)
        ax.add_artist(circle)
        ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)')
        plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18, weight='bold', color='darkblue')
        plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18, weight='bold', color='darkblue')
        ax.figure.set_size_inches(10, 10)
        #plt.grid(False)
        plt.xticks(np.arange(-0.5, 0.75, 0.25))
        plt.yticks(np.arange(-0.5, 0.75, 0.25))

        PSize = (100 / 560)
        rectA = plt.Rectangle(
            (dat['platformPosition_x'].iloc[0] - (PSize / 2), dat['platformPosition_y'].iloc[0] - (PSize / 2)),
            PSize, PSize, linewidth=2.5, edgecolor='yellow',linestyle='--',
            facecolor='none')

        ax.add_artist(rectA)

        plt.tight_layout()


        file_name = f"{Output_Dir}MWM_Heat_Maps_2D/{Mod}/{Bloc}/{Grupo}_Navi_MAP.png"
        plt.savefig(file_name)

        plt.clf()
        print('Mapa de Calor ' + Title + ' Listo')

# Heat Maps!

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']
Group_List = categorias_ordenadas
Mod_List=['No Inmersivo','Realidad Virtual']
Block_List = ['HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']

for B in Block_List:
    for M in Mod_List:
        for G in Group_List:
            MapaDeCalor(df,M,B,G,'MWM Navigation Heat Map ')

print('All right fellas... hope it was worth it...')

#%%