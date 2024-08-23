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

df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
df = df.reset_index()
df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
#df = df.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
df.rename(columns={'Dg': 'Dx'}, inplace=True)
# Función para añadir categorías
def añadir_categorias(fila):
    categorias = []
    if fila['Grupo'] == 'MPPP':
        categorias.append('PPPD')
    if isinstance(fila['Dx'], str) and 'MV' in fila['Dx']:  # Mantener estas líneas si quieres contar 'Vestibular Migraine'
        categorias.append('Vestibular Migraine')
    if fila['Grupo'] == 'Vestibular':
        categorias.append('Vestibular (non PPPD)')
    if fila['Grupo'] == 'Voluntario Sano':
        categorias.append('Healthy Volunteer')
    return categorias

# Lista para almacenar las filas duplicadas
filas_duplicadas = []

# Expandir el DataFrame duplicando las filas según las categorías
for _, fila in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
    categorias = añadir_categorias(fila)
    for categoria in categorias:
        nueva_fila = fila.copy()
        nueva_fila['Categoria'] = categoria
        filas_duplicadas.append(nueva_fila)

# Crear el DataFrame expandido
df_expandido = pd.DataFrame(filas_duplicadas)

# Filtrar los sujetos que son "Vestibular Migraine"
vestibular_migraine_df = df_expandido[df_expandido['Categoria'] == 'Vestibular Migraine']

# Contar el total de sujetos en "Vestibular Migraine"
total_vestibular_migraine = vestibular_migraine_df['Sujeto'].nunique()

# Contar cuántos sujetos del grupo original "MPPP/PPPD" son "Vestibular Migraine"
pppd_vestibular_migraine = vestibular_migraine_df[vestibular_migraine_df['Grupo'] == 'MPPP']['Sujeto'].nunique()

# Contar cuántos sujetos del grupo original "Vestibular non PPPD" son "Vestibular Migraine"
non_pppd_vestibular_migraine = vestibular_migraine_df[vestibular_migraine_df['Grupo'] == 'Vestibular']['Sujeto'].nunique()

# Reportar los resultados
print(f"Total de sujetos en la categoría 'Vestibular Migraine': {total_vestibular_migraine}")
print(f"De estos, {pppd_vestibular_migraine} sujetos son del grupo original 'MPPP/PPPD'.")
print(f"Y {non_pppd_vestibular_migraine} sujetos son del grupo original 'Vestibular non PPPD'.")

#%%
print('final')