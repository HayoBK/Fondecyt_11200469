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

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"

file = Py_Processing_Dir+'DA_EyeTracker_Synced_NI_2D.csv'
df_2D = pd.read_csv(file, index_col=0)

Bloques=['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df_2D= df_2D[df_2D['MWM_Block'].isin(Bloques)]

Codex_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
Codex_df = Codex_df.reset_index()
Codex_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
df_2D = df_2D.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
df_2D.rename(columns={'Dg': 'Dx'}, inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Vamos a añadir Scanned Path de Pedro.-----------------------------------------------------------------------------------------------------
df_2D['delta_x'] = df_2D['x_norm'].diff()
df_2D['delta_y'] = df_2D['y_norm'].diff()

# Calcula la distancia euclidiana
df_2D['distancia'] = np.sqrt(df_2D['delta_x']**2 + df_2D['delta_y']**2)

# Opcionalmente, puedes eliminar las columnas temporales 'delta_x' y 'delta_y'
df_2D = df_2D.drop(columns=['delta_x', 'delta_y'])

# Para la primera fila, donde no hay fila previa, puedes reemplazar NaN con 0 o algún otro valor si lo prefieres
df_2D['distancia'].fillna(0, inplace=True)

#------------------------------------------------------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------------------------------------------------------

# Vamos a Generar duraciones para cada OW_Trial, para asegurar un peso equivalente entre cada OW_a traves de sujetos.......-----------------------------------
#df_whole['timestamp'] = pd.to_datetime(df_whole['timestamp'])

# Agrupamos por '4X-Code' y 'OW_Trial' y calculamos la duración de cada trial
duraciones = df_2D.groupby(['Sujeto', 'OW_Trial'])['timestamp'].agg(Inicio='min', Fin='max')
duraciones['Duracion'] = duraciones['Fin'] - duraciones['Inicio']

# Unimos la información de duración de vuelta al DataFrame original
df_2D = df_2D.merge(duraciones['Duracion'], on=['Sujeto', 'OW_Trial'], how='left')

#------------------------------------------------------------------------------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------------------------------------------------------------------------------
df_2D_filtrado = df_2D.groupby(['Sujeto', 'OW_Trial']).apply(lambda x: x.iloc[1:]).reset_index(drop=True)
df_2D = df_2D[df_2D['on_surf']==True]
#Ahora haremos distancias sumadas, y luego lo haremos por unidad de tiempo para normalizar por trials de distinta duracion...--------------------------------

# Asegúrate de que ya has calculado las distancias y las duraciones como hemos discutido anteriormente

# Paso 1: Sumar las distancias por grupo (4X-Code, OW_Trial)
distancia_sumada = df_2D.groupby(['Sujeto', 'OW_Trial'])['distancia'].sum().reset_index()

# Renombramos la columna para mayor claridad
distancia_sumada.rename(columns={'distancia': 'distancia_sumada'}, inplace=True)

# Unimos la columna de distancia sumada al DataFrame original
df_2D = df_2D.merge(distancia_sumada, on=['Sujeto', 'OW_Trial'], how='left')

# Paso 2: Calcular la "distancia por unidad de tiempo"
df_2D['Scanned Path / time'] = df_2D['distancia_sumada'] / df_2D['Duracion']

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Falto reducir por MWM_Block
#------------------------------------------------------------------------------------------------------------------------------------------------------



promedios = df_2D.groupby(['Sujeto', 'MWM_Block'])['Scanned Path / time'].mean().reset_index()

# Renombramos la columna para mayor claridad
promedios.rename(columns={'Scanned Path / time': 'Scanned_Path_per_time_per_Block'}, inplace=True)

# Ahora puedes unir estos promedios al DataFrame original o trabajar solo con el DataFrame de promedios
df_2D = df_2D.merge(promedios, on=['Sujeto', 'MWM_Block'], how='left')


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

file = Py_Processing_Dir+'DA_EyeTracker_Synced_NI_2D_withVM.csv'
df_2D.to_csv(file)
print("H-Segmento Listo")


#%%
print("Script listo")



#%%
print('God in his heaven, all is right on Earth')