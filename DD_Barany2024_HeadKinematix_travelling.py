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

codex = pd.read_excel((Py_Processing_Dir+'AB_OverWatch_Codex.xlsx'),index_col=0) # Aqui estoy cargando como DataFrame la lista de códigos que voy a usar, osea, los datos del diccionario. Es super
# imporatante el index_col=0 porque determina que la primera columna es el indice del diccionario, el valor que usaremos para guiar los reemplazos.
Codex_Dict = codex.to_dict('series') # Aqui transformo esa Dataframe en una serie, que podré usar como diccionario
df['MWM_Block'] = df['True_OW_Trial'] # Aqui empieza la magia, creo una nueva columna llamada MWM_Bloque, que es una simple copia de la columna OverWatch_Trial.
# De
# momento
# son identicas
df['MWM_Block'].replace(Codex_Dict['MWM_Bloque'], inplace=True) # Y aqui ocurre la magia: estoy reemplazando cada valor de la columna recien creada,
# ocupando el diccionario que
# armamos como guia para hacer el reemplazo
df['MWM_Trial'] = df['True_OW_Trial']
df['MWM_Trial'].replace(Codex_Dict['MWM_Trial'], inplace=True)

codex2 = pd.read_excel((Py_Processing_Dir+'AA_CODEX.xlsx'), index_col=0)
Codex_Dict2 = codex2.to_dict('series')
df['Grupo'] = df['Sujeto']
df['Grupo'].replace(Codex_Dict2['Grupo'], inplace=True)

Bloques=['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
df= df[df['MWM_Block'].isin(Bloques)]

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage3.csv'
df.to_csv(file)

print('Go on')

#%%

promediosvRoll = df.groupby(['Sujeto', 'MWM_Block'])['vRoll_normalizada'].mean().reset_index()
promediosvYaw = df.groupby(['Sujeto', 'MWM_Block'])['vYaw_normalizada'].mean().reset_index()
promediosvPitch = df.groupby(['Sujeto', 'MWM_Block'])['vPitch_normalizada'].mean().reset_index()
promediosAngMagnitud = df.groupby(['Sujeto', 'MWM_Block'])['AngMagnitud_normalizada'].mean().reset_index()

# Renombramos la columna para mayor claridad
promediosvRoll.rename(columns={'vRoll_normalizada': 'vRoll_normalizada_por_Bloque'}, inplace=True)
promediosvYaw.rename(columns={'vYaw_normalizada': 'vYaw_normalizada_por_Bloque'}, inplace=True)
promediosvPitch.rename(columns={'vPitch_normalizada': 'vPitch_normalizada_por_Bloque'}, inplace=True)
promediosAngMagnitud.rename(columns={'AngMagnitud_normalizada': 'AngMagnitud_normalizada_por_Bloque'}, inplace=True)

# Ahora puedes unir estos promedios al DataFrame original o trabajar solo con el DataFrame de promedios
df_ = df.merge(promediosvRoll, on=['Sujeto', 'MWM_Block'], how='left')
df_ = df.merge(promediosvYaw, on=['Sujeto', 'MWM_Block'], how='left')
df_ = df.merge(promediosvPitch, on=['Sujeto', 'MWM_Block'], how='left')
df_ = df.merge(promediosAngMagnitud, on=['Sujeto', 'MWM_Block'], how='left')


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
for _, fila in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
    categorias = añadir_categorias(fila)
    for categoria in categorias:
        nueva_fila = fila.copy()
        nueva_fila['Categoria'] = categoria
        filas_duplicadas.append(nueva_fila)

df = pd.DataFrame(filas_duplicadas)

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage4.csv'
df.to_csv(file)



df['4X-Code']=df['Sujeto']+df['Categoria']
columnas_para_unicas = ['4X-Code', 'MWM_Block']

# Eliminar filas duplicadas basadas en estas columnas
df_reducido = df.drop_duplicates(subset=columnas_para_unicas, keep='first').reset_index(drop=True)

file = Py_Processing_Dir + 'HeadKinematic_Processing_Stage5.csv'
df_reducido.to_csv(file)

print('Se fini...')
#------------------------------------------------------------------------------------------------------------------------------------------------------
# Falto reducir por MWM_Block
#------------------------------------------------------------------------------------------------------------------------------------------------------

