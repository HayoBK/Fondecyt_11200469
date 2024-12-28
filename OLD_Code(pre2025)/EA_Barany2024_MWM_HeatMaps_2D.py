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

df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z2_NaviData_con_posicion.csv'), low_memory=False, index_col=0)
df= df.reset_index(drop=True)
print('Archivo de Navi cargado')
#%%
df['delta_x'] = df['P_position_x'].diff()
df['delta_y'] = df['P_position_y'].diff()
df['distancia'] = np.sqrt(df['delta_x']**2 + df['delta_y']**2)

print('Distancias generadas')

# Opcionalmente, puedes eliminar las columnas temporales 'delta_x' y 'delta_y'
df = df.drop(columns=['delta_x', 'delta_y'])
df = df.groupby(['Sujeto', 'Trial_Unique_ID']).apply(lambda x: x.iloc[1:]).reset_index(drop=True)

print('Algunas distancias iniciales eliminadas')
#%%
df = df[df['distancia'] >= 0.002]  # AQUI Seleccionamos cuales puntos nos quedamos!

print('Limpiado el exceso de distancia')


#%%

print('Invocamos CODEX')
c_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
#c_df = df.reset_index()
#c_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
#%%
Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

df= df[df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]
#%%
#codex = pd.read_excel((Py_Processing_Dir+'AB_OverWatch_Codex.xlsx'),index_col=0) # Aqui estoy cargando como DataFrame la lista de códigos que voy a usar, osea, los datos del diccionario. Es super
# imporatante el index_col=0 porque determina que la primera columna es el indice del diccionario, el valor que usaremos para guiar los reemplazos.
Codex_Dict = c_df.to_dict('series') # Aqui transformo esa Dataframe en una serie, que podré usar como diccionario
df['Dx'] = df['Sujeto'] # Aqui empieza la magia, creo una nueva columna llamada MWM_Bloque, que es una simple copia de la columna OverWatch_Trial.
# De
# momento
# son identicas
df['Dx'].replace(Codex_Dict['Dg'], inplace=True) # Y aqui ocurre la magia: estoy reemplazando cada valor de la columna recien creada,
# ocupando el diccionario que
# armamos como guia para hacer el reemplazo

print('Iniciando Categorizaciòn con MV')
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

print('Ready for the heavy part...')

# Expandir el DataFrame duplicando las filas según las categorías
for _, fila in tqdm(df.iterrows(), total=len(df), desc="Procesando filas"):
    categorias = añadir_categorias(fila)
    for categoria in categorias:
        nueva_fila = fila.copy()
        nueva_fila['Categoria'] = categoria
        filas_duplicadas.append(nueva_fila)
print('Completada la parte HEavy Metal')

df = pd.DataFrame(filas_duplicadas)
file = Py_Processing_Dir + 'EA_MWM_Position_withMV_2D.csv'
df.to_csv(file)
print('GRabado en un CSV independiente el resultado')
#%%

def MapaDeCalor(dat, Mod, Bloc, Grupo, Titulo):
    colores = sns.color_palette('coolwarm',n_colors=100)
    dat = dat.loc[(dat['Grupo'] == Grupo) & (dat['Modalidad'] == Mod) & (dat['True_Block'] == Bloc)]
    Title = str(Titulo) + str(Mod) + '_' + str(Bloc) + '_' + str(Grupo)
    dat.reset_index()
    sns.set_style("ticks")
    sns.set(style='ticks',rc={"axes.facecolor":colores[0]},font_scale=1.5)
    ax = sns.kdeplot(data=dat, x='P_position_x', y='P_position_y', cmap='coolwarm', n_levels=100, thresh=0, fill=True, cbar=True)

    sns.set_context(font_scale=3)
    ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)
    ax.tick_params(labelsize=13)
    ax.set_title(Title, fontsize=22)
    circle = plt.Circle((0, 0), 0.5, color='w',linewidth= 2, fill=False)
    ax.add_artist(circle)
    ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)')
    plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18)
    plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18)
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

    file_name = f"{Output_Dir}MWM_Heat_Maps_2D/Navi_MAP.png"
    plt.savefig(file_name)

    # plt.show()
    plt.clf()
    print('Mapa de Calor ' + Title + ' Listo')


#%%