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
import time

from DA_For_Barany2024 import categorias

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


file = Py_Processing_Dir+'DA_EyeTracker_Synced_NI_2D_withVM.csv'
df_whole = pd.read_csv(file, index_col=0, low_memory=False)
df_whole = df_whole[df_whole['on_surf']==True]

df_whole['4X-Code']=df_whole['Sujeto']+df_whole['Categoria']
df_whole= df_whole[df_whole['MWM_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]

summary_df = df_whole.groupby('4X-Code')['x_norm'].agg(['mean', 'std']).reset_index()

# Merge the Categoria column into the summary_df
summary_df = summary_df.merge(df_whole[['4X-Code', 'Categoria','MWM_Block']].drop_duplicates(), on='4X-Code', how='left')

# Rename columns for clarity
summary_df = summary_df.rename(columns={'mean': 'y_norm_mean', 'std': 'y_norm_std'})


print('Listo primer bloque de calculo')
#%%
Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

for Bl in Bloques_de_Interes:

    data=summary_df[summary_df['MWM_Block'].isin(Bl[1])]
    ax = sns.boxplot(data, x='Categoria', y='y_norm_mean', linewidth=6, order=categorias_ordenadas)
    sns.stripplot(data=data, x='Categoria', y='y_norm_mean', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)

    ax.set_ylabel("Y-Axis Gaze Distribution (Mean position)", weight='bold')
    ax.set_xlabel("Category", weight='bold')
    ax.set_xticklabels(categorias_ordenadas)

    ax.set(ylim=(0, 1))
    #ax.set_title(Title, weight='bold')
    # Determine the y position for the line and annotation
    file_name = f"{Output_Dir}02a - Gaze2D/01X-Summary_Gaze_Mean {Bl[0]}.png"
    plt.savefig(file_name)
    plt.show()
    plt.clf()

    ax = sns.boxplot(data, x='Categoria', y='y_norm_std', linewidth=6, order=categorias_ordenadas)
    sns.stripplot(data=data, x='Categoria', y='y_norm_std', jitter=True, color='black', size=10, ax=ax,
                  order=categorias_ordenadas)

    ax.set_ylabel("Y-Axis Gaze Distribution (Mean position)", weight='bold')
    ax.set_xlabel("Category", weight='bold')
    ax.set_xticklabels(categorias_ordenadas)

    ax.set(ylim=(0, 1))
    # ax.set_title(Title, weight='bold')
    # Determine the y position for the line and annotation
    file_name = f"{Output_Dir}02a - Gaze2D/02X-Summary_Gaze_Std {Bl[0]}.png"
    plt.savefig(file_name)
    plt.show()
    plt.clf()

#%%



def remove_extreme_timestamps(group):
    # Sort by timestamp just in case it's not sorted
    group = group.sort_values(by='timestamp')

    # Calculate the indices for the 20% and 80% thresholds
    lower_index = int(len(group) * 0.5)
    upper_index = int(len(group) * 0.85)

    # Keep only the middle 60% of the data
    return group.iloc[lower_index:upper_index]

def normalize_trial_length(df, n_points=5000):
    # Create a new time axis with n_points
    df_normalized = df.groupby(['Sujeto', 'Categoria', 'OW_Trial']).apply(
        lambda x: x.set_index('timestamp').reindex(
            np.linspace(x['timestamp'].min(), x['timestamp'].max(), n_points)
        ).interpolate().reset_index(drop=True)
    )
    return df_normalized

#df_whole = normalize_trial_length(df_whole)

processed_data = []
grouped = df_whole.groupby(['Sujeto', 'Categoria'])
target_sample_size = 20000


for (sujeto, categoria), group in grouped:
    print(f'Pre-Processing Sujeto: {sujeto}, Categoria: {categoria}')

    # Optionally filter x_norm and y_norm to be within -3 and 3
    group = group[(group['x_norm'] >= -3) & (group['x_norm'] <= 3) &
                  (group['y_norm'] >= -3) & (group['y_norm'] <= 3)]
    group = group.groupby('OW_Trial').apply(remove_extreme_timestamps).reset_index(drop=True)

    # Downsample the group to the target sample size
    if group.shape[0] > target_sample_size:
        group = group.sample(target_sample_size, random_state=42)

    # Append the processed group to the list
    processed_data.append(group)

# Combine all processed groups back into a single DataFrame
df_whole = pd.concat(processed_data)

# Apply the function to each trial

categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

Bloques_de_Interes = []
Bloques_de_Interes.append(['HT_1',['HiddenTarget_1']])
Bloques_de_Interes.append(['HT_2',['HiddenTarget_2']])
Bloques_de_Interes.append(['HT_3',['HiddenTarget_3']])
Bloques_de_Interes.append(['All_HT',['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']])

sujetos_unicos = df_whole['Sujeto'].unique()
print(sujetos_unicos)

for Bl in Bloques_de_Interes:


    df= df_whole[df_whole['MWM_Block'].isin(Bl[1])]


    for categoria in categorias_ordenadas:
        print('Iniciando Procesamiento de ', Bl[0], categoria)

        for S in sujetos_unicos:
            data = df[df['Categoria'] == categoria]
            data=data[data['Sujeto']==S]
            data = data[['x_norm', 'y_norm']]
            #print(data.shape[0])
            if data.shape[0] > 1000:  # Downsample if necessary
                data = data.sample(1000, random_state=42)
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
                file_name = f"{Output_Dir}02a - Gaze2D/S/{categoria} {S} {Bl[0]}_Gaze_Distribution.png"
                plt.savefig(file_name)
                plt.clf()

        inicio_bloque = time.time()
        data = df[df['Categoria']==categoria]
        data=data[['x_norm', 'y_norm']]
        #print(data.shape[0])
        if data.shape[0] > 50000:  # Downsample if necessary
            data = data.sample(50000, random_state=42)
        #print(data.shape[0])
        #print('sub-df generada')
    # Recorremos las categorías desde la lista predefinida
        print(f'Generando gráfico MAYOR para la categoría: {categoria} en {Bl[0]}')
        sns.kdeplot(data=data, x='x_norm', y='y_norm', cmap='coolwarm', n_levels=100, thresh=0, fill=True, cbar=True)
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
        file_name = f"{Output_Dir}02a - Gaze2D/{Bl[0]}_Gaze_Distribution_{categoria}_.png"
        plt.savefig(file_name)
        plt.clf()
        duracion = time.time() - inicio_bloque
        print(f'Terminando Procesamiento de {categoria} {Bl[0]} en {duracion:.2f} segundos')


    print('Preparando Boxplot')
    plt.figure(figsize=(12, 8))
    sns.boxplot(y='y_norm', x='Categoria', data=df, order=categorias_ordenadas,
                palette='Set2')
    plt.ylabel('Y-Axis Gaze Distribution')
    plt.xlabel('Group / Diagnosis')
    Title = 'Y-Axis Gaze Distribution ' + Bl[0]
    plt.title(Title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.savefig(f"{Output_Dir}02a - Gaze2D/Y-Axis Gaze Distribution {Bl[0]}.png")
    plt.clf()

print('Segmento: God in his heaven, all is right on Earth')

#%%
print('God in his heaven, all is right on Earth')