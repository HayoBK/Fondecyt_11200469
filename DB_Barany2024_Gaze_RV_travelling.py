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

file = Py_Processing_Dir+'DA_EyeTracker_Synced_RV_3D_withVM.csv'
df_whole = pd.read_csv(file, index_col=0, low_memory=False)
df_whole = df_whole.copy()
df_whole = df_whole[~df_whole['Sujeto'].isin(['P05','P14','P16'])]
#%%
df_whole['4X-Code']=df_whole['Sujeto']+df_whole['Categoria'] # Añadimos un codigo de Sujeto unico nuevo para evitar sujetos repetidos por la asignación de la
                                                                # Categoria MV
Esto_si = False
if Esto_si:
    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    #Vamos a añadir Scanned Path de Pedro.-----------------------------------------------------------------------------------------------------
    df_whole['delta_x'] = df_whole['norm_pos_x'].diff()
    df_whole['delta_y'] = df_whole['norm_pos_y'].diff()

    # Calcula la distancia euclidiana
    df_whole['distancia'] = np.sqrt(df_whole['delta_x']**2 + df_whole['delta_y']**2)

    # Opcionalmente, puedes eliminar las columnas temporales 'delta_x' y 'delta_y'
    df_whole = df_whole.drop(columns=['delta_x', 'delta_y'])

    # Para la primera fila, donde no hay fila previa, puedes reemplazar NaN con 0 o algún otro valor si lo prefieres
    df_whole['distancia'].fillna(0, inplace=True)

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    # Vamos a Generar duraciones para cada OW_Trial, para asegurar un peso equivalente entre cada OW_a traves de sujetos.......-----------------------------------
    #df_whole['timestamp'] = pd.to_datetime(df_whole['timestamp'])

    # Agrupamos por '4X-Code' y 'OW_Trial' y calculamos la duración de cada trial
    duraciones = df_whole.groupby(['4X-Code', 'OW_Trial'])['timestamp'].agg(Inicio='min', Fin='max')
    duraciones['Duracion'] = duraciones['Fin'] - duraciones['Inicio']

    # Unimos la información de duración de vuelta al DataFrame original
    df_whole = df_whole.merge(duraciones['Duracion'], on=['4X-Code', 'OW_Trial'], how='left')

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    #
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    #Ahora haremos distancias sumadas, y luego lo haremos por unidad de tiempo para normalizar por trials de distinta duracion...--------------------------------

    # Asegúrate de que ya has calculado las distancias y las duraciones como hemos discutido anteriormente

    # Paso 1: Sumar las distancias por grupo (4X-Code, OW_Trial)
    distancia_sumada = df_whole.groupby(['4X-Code', 'OW_Trial'])['distancia'].sum().reset_index()

    # Renombramos la columna para mayor claridad
    distancia_sumada.rename(columns={'distancia': 'distancia_sumada'}, inplace=True)

    # Unimos la columna de distancia sumada al DataFrame original
    df_whole = df_whole.merge(distancia_sumada, on=['4X-Code', 'OW_Trial'], how='left')

    # Paso 2: Calcular la "distancia por unidad de tiempo"
    df_whole['Scanned Path / time'] = df_whole['distancia_sumada'] / df_whole['Duracion']

    #------------------------------------------------------------------------------------------------------------------------------------------------------
    # Falto reducir por MWM_Block
    #------------------------------------------------------------------------------------------------------------------------------------------------------



    promedios = df_whole.groupby(['4X-Code', 'MWM_Block'])['Scanned Path / time'].mean().reset_index()

    # Renombramos la columna para mayor claridad
    promedios.rename(columns={'Scanned Path / time': 'Scanned_Path_per_time_per_Block'}, inplace=True)

    # Ahora puedes unir estos promedios al DataFrame original o trabajar solo con el DataFrame de promedios
    df_whole = df_whole.merge(promedios, on=['4X-Code', 'MWM_Block'], how='left')


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Vamonos a graficar
#------------------------------------------------------------------------------------------------------------------------------------------------------


Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

# Seleccionar las columnas relevantes para identificar duplicados
columnas_para_unicas = ['4X-Code', 'MWM_Block']

# Eliminar filas duplicadas basadas en estas columnas
df_reducido = df_whole.drop_duplicates(subset=columnas_para_unicas, keep='first').reset_index(drop=True)

columnas_para_unicas = ['4X-Code', 'OW_Trial']

# Eliminar filas duplicadas basadas en estas columnas
df_read = df_whole.drop_duplicates(subset=columnas_para_unicas, keep='first').reset_index(drop=True)

df_reducido['4X-Code'] = df_reducido['4X-Code'].str[:6]

custom_palette = sns.color_palette(["#ADD8E6", "#DDA0DD", "#FFA07A", "#98FB98"])  # Light blue, light purple, light orange, light green


for Bl in Bloques_de_Interes:
    if Bl[1]:
        data=df_reducido[df_reducido['MWM_Block'].isin(Bl[1])]
    else:
        data=df_reducido
    file = Py_Processing_Dir + 'DA_Gaze_RV_reducido_'+Bl[0]+'.csv'
    data.to_csv(file)
    print(f"Generando Grafico para {Bl[0]}")
    fig, ax = plt.subplots(figsize=(10, 8))
    custom_palette = sns.color_palette(["#ADD8E6", "#DDA0DD", "#FFA07A", "#98FB98"])
    color_mapping = {
        'PPPD': "#FFA07A",
        'Vestibular Migraine': "#DDA0DD",
        'Vestibular (non PPPD)': "#ADD8E6",
        'Healthy Volunteer': "#98FB98"
    }
    custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]

    ax = sns.boxplot(data, x='Categoria', y='Scanned_Path_per_time_per_Block', linewidth=6, order=categorias_ordenadas, hue='Categoria', legend=False, palette=custom_palette)
    sns.stripplot(data=data, x='Categoria', y='Scanned_Path_per_time_per_Block', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
    offsets = ax.collections[-1].get_offsets()
    #for i, (x, y) in enumerate(offsets):
    #    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
    #                ha='center', va='center', fontsize=8, color='black')
    ax.set_ylabel("Gaze Scanned Path / Time ", weight='bold')
    ax.set_xlabel("Category", weight='bold')
    ax.set_xticks(range(len(categorias_ordenadas)))
    ax.set_xticklabels(categorias_ordenadas)
    #ax.get_legend().remove()
    ax.set(ylim=(0, 350))
    #ax.set_title(Title, weight='bold')
    # Determine the y position for the line and annotation
    file_name = f"{Output_Dir}RV_AllNew_{Bl[0]}_Gaze_Scanned_Path.png"
    plt.savefig(file_name)
    plt.clf()
    print(f"--Completo Grafico para {Bl[0]}")

#---------------------------------------------------------------------------------------------------
df_whole = df_whole[df_whole['distancia'] >= 0.05]  # AQUI Seleccionamos cuales puntos nos quedamos!
#---------------------------------------------------------------------------------------------------

sujetos_unicos = df_whole['4X-Code'].unique()
print(sujetos_unicos)

Bloques_de_Interes = []
#Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
#Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
#Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])


for Bl in Bloques_de_Interes:

    df= df_whole[df_whole['MWM_Block'].isin(Bl[1])]

    for categoria in categorias_ordenadas:
        print('Iniciando Procesamiento de ', Bl[0], categoria)

        GoNoGo = False
        if GoNoGo:
            for S in sujetos_unicos:
                data = df[df['Categoria'] == categoria]
                data=data[data['4X-Code']==S]
                data = data[['x_norm', 'y_norm']]
                #print(data.shape[0])
                if data.shape[0] > 10000:  # Downsample if necessary
                    data = data.sample(10000, random_state=42)
                #print(data.shape[0])
                #print('sub-df generada')
                if data.shape[0] > 0:
                    # Recorremos las categorías desde la lista predefinida
                    print(f'Generando gráfico para la categoría: {categoria} en {Bl[0]}')
                    sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=100, thresh=0, fill=True,
                                cbar=True)
                    plt.xlim(0, 1)
                    plt.ylim(0, 1)

                    # Añadimos líneas de guía
                    for val in [0.25, 0.75]:
                        plt.axvline(val, color='gray', linestyle='--', lw=1)
                        plt.axhline(val, color='gray', linestyle='--', lw=1)
                    plt.axvline(0.5, color='gray', linestyle='--', lw=2)
                    plt.axhline(0.5, color='gray', linestyle='--', lw=2)

                    # Añadimos el título
                    plt.title(f'Gaze Distribution for {categoria} in {Bl[0]}')

                    # Guardamos el gráfico
                    file_name = f"{Output_Dir}2Dper4X-Code/{categoria} {S} {Bl[0]}_Gaze_Distribution.png"
                    plt.savefig(file_name)
                    plt.clf()

        inicio_bloque = time.time()
        data = df[df['Categoria']==categoria]
        data=data[['norm_pos_x', 'norm_pos_y']]
        #print(data.shape[0])
        if data.shape[0] > 700:  # Downsample if necessary
            data = data.sample(700, random_state=42)
        #print(data.shape[0])
        #print('sub-df generada')
    # Recorremos las categorías desde la lista predefinida
        print(f'Generando gráfico MAYOR para la categoría: {categoria} en {Bl[0]}')
        #sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=300, thresh=0, fill=True, cbar=True)
        plt.figure(figsize=(10, 8))

        # Crear el jointplot sin la barra de color
        g = sns.jointplot(
            data=data,
            x='norm_pos_x',
            y='norm_pos_y',
            kind='kde',
            fill=True,
            cmap='viridis',
            thresh=0,
            levels=25,
            cbar=False,
            marginal_kws={'shade': True}
        )

        # Añadir la barra de color manualmente
        #plt.colorbar(g.ax_joint.collections[0], ax=g.ax_joint, location="right", pad=0.02)

        # Ajustar límites de los ejes
        g.ax_joint.set_xlim(-2, 3)
        g.ax_joint.set_ylim(-2, 3)

        # Añadir líneas de guía en el gráfico central
        for val in [0.25, 0.75]:
            g.ax_joint.axvline(val, color='gray', linestyle='--', lw=1)
            g.ax_joint.axhline(val, color='gray', linestyle='--', lw=1)
        g.ax_joint.axvline(0.5, color='gray', linestyle='--', lw=2)
        g.ax_joint.axhline(0.5, color='gray', linestyle='--', lw=2)

        # Añadir el título y ajustar su posición
        g.fig.suptitle(f'{categoria} - Gaze Distribution over the Screen',
                       fontsize=18, weight='bold', color='darkblue', y=1.03)

        # Añadir etiquetas a los ejes y ajustar el padding
        g.set_axis_labels('Normalized X Position', 'Normalized Y Position',
                          fontsize=14, weight='bold', color='darkred', labelpad=15)

        #g.ax_joint.set_aspect(aspect=(1 / 1.78))
        # Mejorar el layout para que todo se ajuste correctamente
        plt.tight_layout()

        # Guardar el gráfico
        file_name = f"{Output_Dir}RV - Mapas de Calor/Final_{Bl[0]}_RV_Gaze_Distribution_{categoria}_.png"
        plt.savefig(file_name, bbox_inches='tight')  # bbox_inches='tight' asegura que los textos no se corten
        plt.clf()
        duracion = time.time() - inicio_bloque
        print(f'Terminando Procesamiento de {categoria} {Bl[0]} en {duracion:.2f} segundos')

print('And Dream of the Endless asked: What do you dream of then, when you dream of sex?')
#%%
print('Final real del archivo')
