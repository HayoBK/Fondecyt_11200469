#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 10 dias.
#
#-----------------------------------------------------

#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import socket


# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------

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

# ------------------------------------------------------------
# Cargando bases de datos de Navegación MWM
# ------------------------------------------------------------

NaviCSE_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
NaviPOS_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z2_NaviData_con_posicion.csv'), low_memory=False, index_col=0)
# Aqui Barany Codex le falta actualización... antes era Fenrir Codex... Para la publiación
# Y uso de dato neurocognitivos, debemos trabajar con OMEGA CODEX
Codex_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
Codex_df = Codex_df.reset_index()
Codex_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
NaviCSE_df = NaviCSE_df.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
NaviPOS_df = NaviPOS_df.merge(Codex_df[['Sujeto', 'Dg']], on='Sujeto', how='left')
NaviCSE_df.rename(columns={'Dg_y': 'Dx'}, inplace=True)
NaviPOS_df.rename(columns={'Dg_y': 'Dx'}, inplace=True)
NaviCSE_df = NaviCSE_df[NaviCSE_df['Sujeto'] != 'P13']
NaviPOS_df = NaviPOS_df[NaviPOS_df['Sujeto'] != 'P13']
NaviCSE_df['Modalidad'] = NaviCSE_df['Modalidad'].replace('No Inmersivo', 'Non-Immersive')
NaviCSE_df['Modalidad'] = NaviCSE_df['Modalidad'].replace('Realidad Virtual', 'Virtual Reality')
NaviPOS_df['Modalidad'] = NaviPOS_df['Modalidad'].replace('No Inmersivo', 'Non-Immersive')
NaviPOS_df['Modalidad'] = NaviPOS_df['Modalidad'].replace('Realidad Virtual', 'Virtual Reality')


#------------------------------------------------------------
#Añadiendo Normalización por ture Block y Modalidad
#------------------------------------------------------------

df = NaviCSE_df
df = df[df['Sujeto'] != 'P13']
#df = df[df['Sujeto'] != 'P07']
group_stats = df.groupby(['True_Block', 'Modalidad']).agg({
    'CSE': ['mean', 'std']
}).reset_index()

# Renombrar columnas para facilitar el acceso
group_stats.columns = ['True_Block', 'Modalidad', 'Mean_CSE', 'Std_CSE']

# Unir las estadísticas de vuelta al DataFrame original
df = df.merge(group_stats, on=['True_Block', 'Modalidad'])

# Normalizar los valores de CSE
df['Norm_CSE'] = (df['CSE'] - df['Mean_CSE']) / df['Std_CSE']
NaviCSE_df = df

#---------------------------------------------------------------------------------
#
#           GRAFICO 1 BARANY 2024
#
# Vamos a Por grafico 1 Barany : Rendimiento CSE; incluyendo Migraña vestibular
#---------------------------------------------------------------------------------

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])
for Bl in Bloques_de_Interes:
    df = NaviCSE_df
    df= df[df['True_Block'].isin(Bl[1])]
    #df= df[df['True_Block'].isin(['HiddenTarget_1'])]


    df_expandido = pd.DataFrame(columns=df.columns)
    filas_duplicadas = []

    # Función para añadir categorías
    def añadir_categorias(fila):
        categorias = []
        if fila['Grupo'] == 'MPPP':
            categorias.append('PPPD')
        if isinstance(fila['Dx'], str) and 'MV' in fila['Dx']:      # Tengo que borrar estas dos lineas si quiero
            categorias.append('Vestibular Migraine')                # Eliminar Migraña vestibular
        if fila['Grupo'] == 'Vestibular':
            categorias.append('Vestibular (non PPPD)')
        if fila['Grupo'] == 'Voluntario Sano':
            categorias.append('Healthy Volunteer')
        return categorias

    # Expandir el DataFrame duplicando las filas según las categorías
    for _, fila in df.iterrows():
        categorias = añadir_categorias(fila)
        for categoria in categorias:
            nueva_fila = fila.copy()
            nueva_fila['Categoria'] = categoria
            filas_duplicadas.append(nueva_fila)

    df_expandido = pd.DataFrame(filas_duplicadas)
    categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']
    #categorias_ordenadas = ['PPPD', 'Vestibular (non PPPD)', 'Healthy Volunteer'] #Tengo que poner esta linea si quiero eliminar migraña vestibular

    # Crear el boxplot
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='Norm_CSE', x='Categoria', hue='Modalidad', data=df_expandido, order=categorias_ordenadas, palette='Set2')
    plt.ylabel('Normalized Cummulative Search Error (CSE)')
    plt.xlabel('Group / Diagnosis')
    Title = '01-Spatial Navigation Impairment '+ Bl[0]
    plt.title(Title)
    plt.legend(title='Modalidad', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(-3, 4)
    plt.savefig(Output_Dir + '01 - Morris/'+ Title + '.png')
    plt.clf()
    #plt.show()




print('Everything LOADED')
print('Ready...')
#%%
print('THE END')