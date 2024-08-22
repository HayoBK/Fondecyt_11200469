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
#           Procesando datos
#----------------------------------------------------------------------------------------------------------

file = Py_Processing_Dir+'HeadKinematic_Raw_v2.csv'
df = pd.read_csv(file, index_col=0, low_memory=False)

Bloques=['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']

Codex_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
Codex_df = Codex_df.reset_index()
Codex_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
df = df.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
df.rename(columns={'Dg': 'Dx'}, inplace=True)
dfback = df.copy()
print('Go on...')

#%% jijijnji

df=dfback

df['delta_vRoll'] = df['vRoll'].diff().abs()
df['delta_vYaw'] = df['vJaw'].diff().abs()
df['delta_vPitch'] = df['vPitch'].diff().abs()

df['rotation_magnitude'] = np.sqrt(df['delta_vYaw']**2 + df['delta_vPitch']**2 + df['delta_vRoll']**2)

df.dropna(subset=['True_OW_Trial'], inplace=True)

duraciones = df.groupby(['Sujeto', 'Modalidad', 'True_OW_Trial'])['TimeStamp'].agg(Inicio='min', Fin='max')
duraciones['Duracion'] = duraciones['Fin'] - duraciones['Inicio']
df= df.merge(duraciones['Duracion'], on=['Sujeto', 'Modalidad', 'True_OW_Trial'], how='left')
df = df.groupby(['Sujeto', 'Modalidad','True_OW_Trial']).apply(lambda x: x.iloc[1:]).reset_index(drop=True) #Elimino todas las primeras filas

delta_vRoll_sumada = df.groupby(['Sujeto', 'Modalidad', 'True_OW_Trial'])['delta_vRoll'].sum().reset_index()
delta_vYaw_sumada = df.groupby(['Sujeto', 'Modalidad', 'True_OW_Trial'])['delta_vYaw'].sum().reset_index()
delta_vPitch_sumada = df.groupby(['Sujeto', 'Modalidad', 'True_OW_Trial'])['delta_vPitch'].sum().reset_index()
delta_AngMagnitud_sumada = df.groupby(['Sujeto', 'Modalidad', 'True_OW_Trial'])['rotation_magnitude'].sum().reset_index()

# Renombramos la columna para mayor claridad
delta_vRoll_sumada.rename(columns={'delta_vRoll': 'vRoll_sumada'}, inplace=True)
delta_vYaw_sumada.rename(columns={'delta_vYaw': 'vYaw_sumada'}, inplace=True)
delta_vPitch_sumada.rename(columns={'delta_vPitch': 'vPitch_sumada'}, inplace=True)
delta_AngMagnitud_sumada.rename(columns={'rotation_magnitude': 'AngMagnitud_sumada'}, inplace=True)

# Unimos la columna de distancia sumada al DataFrame original
df = df.merge(delta_vRoll_sumada, on=['Sujeto', 'Modalidad','True_OW_Trial'], how='left')
df = df.merge(delta_vYaw_sumada, on=['Sujeto', 'Modalidad','True_OW_Trial'], how='left')
df = df.merge(delta_vPitch_sumada, on=['Sujeto', 'Modalidad','True_OW_Trial'], how='left')
df = df.merge(delta_AngMagnitud_sumada, on=['Sujeto', 'Modalidad','True_OW_Trial'], how='left')

# Paso 2: Calcular la "distancia por unidad de tiempo"
df['vRoll_normalizada'] = df['vRoll_sumada'] / df['Duracion']
df['vYaw_normalizada'] = df['vYaw_sumada'] / df['Duracion']
df['vPitch_normalizada'] = df['vPitch_sumada'] / df['Duracion']
df['AngMagnitud_normalizada'] = df['AngMagnitud_sumada'] / df['Duracion']

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage2.csv'
df.to_csv(file)

print('Go on')

#%%

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Falto reducir por MWM_Block
#------------------------------------------------------------------------------------------------------------------------------------------------------

