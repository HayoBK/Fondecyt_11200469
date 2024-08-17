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

file = Py_Processing_Dir+'DA_EyeTracker_Synced_RV_3D.csv'
df_2D = pd.read_csv(file, index_col=0)

Bloques=['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df_2D= df_2D[df_2D['MWM_Block'].isin(Bloques)]

Codex_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
Codex_df = Codex_df.reset_index()
Codex_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
df_2D = df_2D.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
df_2D.rename(columns={'Dg': 'Dx'}, inplace=True)

filas_duplicadas = []
def añadir_categorias(fila):
    categorias = []
    if fila['Grupo'] == 'MPPP':
        categorias.append('PPPD')
    if isinstance(fila['Dx'], str) and 'MV' in fila['Dx']:  # Tengo que borrar estas dos lineas si quiero
        categorias.append('Vestibular Migraine')  # Eliminar Migraña vestibular
    if fila['Grupo'] == 'Vestibular':
        categorias.append('Vestibular (non PPPD)')
    if fila['Grupo'] == 'Voluntario Sano':
        categorias.append('Healthy Volunteer')
    return categorias


# Expandir el DataFrame duplicando las filas según las categorías
for _, fila in tqdm(df_2D.iterrows(), total=len(df_2D), desc="Procesando filas"):
    categorias = añadir_categorias(fila)
    for categoria in categorias:
        nueva_fila = fila.copy()
        nueva_fila['Categoria'] = categoria
        filas_duplicadas.append(nueva_fila)

df_2D = pd.DataFrame(filas_duplicadas)

file = Py_Processing_Dir+'DA_EyeTracker_Synced_RV_3D_withVM.csv'
df_2D.to_csv(file)
print("H-Segmento Listo")


#%%
print("Script listo")



#%%
print('God in his heaven, all is right on Earth')