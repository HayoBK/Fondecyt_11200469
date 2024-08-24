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

df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), low_memory=False, index_col=0)
df= df.reset_index(drop=True)
print('Archivo de Navi cargado')
df= df[df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]
df = df.groupby(['Sujeto','Modalidad','True_Block'])['CSE'].agg(['mean']).reset_index()


#%%

print('Invocamos CODEX')
c_df = pd.read_excel((Py_Processing_Dir+'BARANY_CODEX.xlsx'), index_col=0)
Codex_Dict = c_df.to_dict('series') # Aqui transformo esa Dataframe en una serie, que podré usar como diccionario
df['Dx'] = df['Sujeto'] # Aqui empieza la magia, creo una nueva columna llamada MWM_Bloque, que es una simple copia de la columna OverWatch_Trial.
df['Dx'].replace(Codex_Dict['Dg'], inplace=True)
df['Grupo'] = df['Sujeto']
df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)

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
color_mapping = {
    'PPPD': "#ADD8E6",
    'Vestibular Migraine': "#DDA0DD",
    'Vestibular (non PPPD)': "#FFA07A" ,
    'Healthy Volunteer': "#98FB98"
}
categorias_ordenadas = ['PPPD', 'Vestibular Migraine', 'Vestibular (non PPPD)', 'Healthy Volunteer']

Bloques_de_Interes=['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
Modalidades = ['No Inmersivo','Realidad Virtual']
for M in Modalidades:
    for Bl in Bloques_de_Interes:
        if Bl:
            data=df[df['True_Block'].isin([Bl])]
            data = data[data['Modalidad'].isin([M])]
        else:
            data=df
        print(f"Generando Grafico para {M} y {Bl}")
        fig, ax = plt.subplots(figsize=(10, 8))
        custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
        ax = sns.boxplot(data=data, x='Categoria', y='mean', linewidth=6, order=categorias_ordenadas, palette=custom_palette)
        sns.stripplot(data=data, x='Categoria', y='mean', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
        offsets = ax.collections[-1].get_offsets()
        #for i, (x, y) in enumerate(offsets):
        #    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
        #                ha='center', va='center', fontsize=8, color='black')
        ax.set_ylabel("Cummulative Search Error - CSE (mean)", fontsize=18, weight='bold', color='darkblue')
        ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')
        #ax.set_xticks(range(len(categorias_ordenadas)))
        #ax.set_xticklabels(categorias_ordenadas)
        #ax.get_legend().remove()
        #ax.set(ylim=(0, 50))
        #if idx == 3:
        #    ax.set(ylim=(0, 100))
        Title = f"CSE (mean) for {Bl} at {M}"
        ax.set_title(Title, fontsize=18, weight='bold', color='darkblue')
        # Determine the y position for the line and annotation
        file_name = f"{Output_Dir}CSE/{M}_{Bl}_CSE.png"
        plt.savefig(file_name)
        plt.clf()
        print(f"--Completo Grafico para {M} & {Bl}")

#%%

df.rename(columns={'mean': 'CSE_mean'}, inplace=True)
group_stats = df.groupby(['True_Block', 'Modalidad']).agg({'CSE_mean': ['mean', 'std']}).reset_index()

# Renombrar columnas para facilitar el acceso
group_stats.columns = ['True_Block', 'Modalidad', 'Mean_CSE', 'Std_CSE']

# Unir las estadísticas de vuelta al DataFrame original
df = df.merge(group_stats, on=['True_Block', 'Modalidad'])

# Normalizar los valores de CSE
df['Norm_CSE'] = (df['CSE_mean'] - df['Mean_CSE']) / df['Std_CSE']

df = df[df['Sujeto'] != 'P13']
df = df[df['Sujeto'] != 'P05']

df['4X-Code'] = df['Sujeto']+df['Categoria']
df = df.groupby(['4X-Code', 'Modalidad', 'Categoria'])['Norm_CSE'].mean().reset_index()


color_mapping = {
    'PPPD': "#ADD8E6",
    'Vestibular Migraine': "#DDA0DD",
    'Vestibular (non PPPD)': "#FFA07A" ,
    'Healthy Volunteer': "#98FB98"
}

data = df[df['Modalidad'].isin(['No Inmersivo'])]
fig, ax = plt.subplots(figsize=(10, 8))
custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]

ax = sns.boxplot(data=data, x='Categoria', y='Norm_CSE', linewidth=6, order=categorias_ordenadas, palette=custom_palette)
sns.stripplot(data=data, x='Categoria', y='Norm_CSE', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
offsets = ax.collections[-1].get_offsets()
#for i, (x, y) in enumerate(offsets):
#    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
#                ha='center', va='center', fontsize=8, color='black')
ax.set_ylabel("Cummulative Search Error - CSE (normalized)", fontsize=18, weight='bold', color='darkblue')
ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')
#ax.set_xticks(range(len(categorias_ordenadas)))
#ax.set_xticklabels(categorias_ordenadas)
#ax.get_legend().remove()
#ax.set(ylim=(0, 50))
#if idx == 3:
#    ax.set(ylim=(0, 100))
Title = f"Navigation Impairment (Non-Immersive)"
ax.set_title(Title, fontsize=18, weight='bold', color='darkblue')
# Determine the y position for the line and annotation
file_name = f"{Output_Dir}CSE/Normalized_CSE_2D.png"
plt.savefig(file_name)
plt.clf()
print(f"--Completo Grafico de Resumen")

data = df
fig, ax = plt.subplots(figsize=(10, 8))
custom_palette = [color_mapping[cat] for cat in categorias_ordenadas]
ax = sns.boxplot(data=data, x='Categoria', y='Norm_CSE', hue='Modalidad', linewidth=6, order=categorias_ordenadas, palette=custom_palette)
sns.stripplot(data=data, x='Categoria', y='Norm_CSE', jitter=True, color='black', size=10, ax=ax, order=categorias_ordenadas)
offsets = ax.collections[-1].get_offsets()
#for i, (x, y) in enumerate(offsets):
#    ax.annotate(data.iloc[i]['4X-Code'], (x, y),
#                ha='center', va='center', fontsize=8, color='black')
ax.set_ylabel("Cummulative Search Error - CSE (normalized)", fontsize=18, weight='bold', color='darkblue')
ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')
#ax.set_xticks(range(len(categorias_ordenadas)))
#ax.set_xticklabels(categorias_ordenadas)
#ax.get_legend().remove()
#ax.set(ylim=(0, 50))
#if idx == 3:
#    ax.set(ylim=(0, 100))
Title = f"Navigation Impairment (Non-Immersive vs Virtual Reality)"
ax.set_title(Title, fontsize=18, weight='bold', color='darkblue')
# Determine the y position for the line and annotation
file_name = f"{Output_Dir}CSE/Normalized_CSE_2DyRV.png"
plt.savefig(file_name)
plt.clf()
print(f"--Completo Grafico de Resumen")

# Data filtering
data = df

# Custom palette with correct order of colors
custom_palette = [
    "#ADD8E6", "#92BCD1",  # Light Blue, Darker Blue
    "#DDA0DD", "#BB8ABD",  # Light Purple, Darker Purple
    "#FFA07A", "#E08A6D",  # Light Salmon, Darker Salmon
    "#98FB98", "#82D782"   # Light Green, Darker Green
]

# Define manual mappings for color_idx and hatches
color_mapping = {
    0: 6,#0,  # Light Blue for first patch (PPPD, No Inmersivo)
    1: 0,  # Darker Blue for second patch (PPPD, Realidad Virtual)
    2: 1,  # Light Purple for third patch (Vestibular Migraine, No Inmersivo)
    3: 1,  # Darker Purple for fourth patch (Vestibular Migraine, Realidad Virtual)
    4: 2,  # Light Salmon for fifth patch (Vestibular non PPPD, No Inmersivo)
    5: 3,  # Darker Salmon for sixth patch (Vestibular non PPPD, Realidad Virtual)
    6: 4,  # Light Green for seventh patch (Healthy Volunteer, No Inmersivo)
    7: 5,  # Darker Green for eighth patch (Healthy Volunteer, Realidad Virtual)
    8: 6,  # Light Blue for ninth patch (PPPD, No Inmersivo - Repeated)
    9: 7   # Darker Blue for tenth patch (PPPD, Realidad Virtual - Repeated)
}

hatch_mapping = {
    0: '',   # No hatch for first patch (No Inmersivo)
    1: '', # Hatch for second patch (Realidad Virtual)
    2: '',   # No hatch for third patch (No Inmersivo)
    3: '//', # Hatch for fourth patch (Realidad Virtual)
    4: '',   # No hatch for fifth patch (No Inmersivo)
    5: '//', # Hatch for sixth patch (Realidad Virtual)
    6: '',   # No hatch for seventh patch (No Inmersivo)
    7: '//', # Hatch for eighth patch (Realidad Virtual)
    8: '',   # No hatch for ninth patch (No Inmersivo - Repeated)
    9: '//'  # Hatch for tenth patch (Realidad Virtual - Repeated)
}

# Plot the boxplot with hue
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=data, x='Categoria', y='Norm_CSE', hue='Modalidad', linewidth=6, order=categorias_ordenadas, ax=ax)

# Apply colors and hatches manually
for i, patch in enumerate(ax.patches):
    color_idx = color_mapping[i]  # Fetch from the manual color mapping
    hatch_idx = i % 2             # Alternating hatches based on manual mapping

    # Apply the color and hatch
    patch.set_facecolor(custom_palette[color_idx])
    patch.set_hatch(hatch_mapping[i])  # Apply the hatch based on manual mapping

# Remove the legend
ax.legend().remove()
ax.set_xticklabels(['PPPD\n ', 'Vestibular\nMigraine', 'Vestibular\n(non PPPD)', 'Healthy\nVolunteer'], weight='bold', fontsize=14)
ax.tick_params(axis='x', labelsize=14)  # x-axis labels

# Set labels and title
ax.set_ylabel("Cummulative Search Error - CSE (normalized)", fontsize=18, weight='bold', color='darkblue')
ax.set_xlabel("Category", fontsize=18, weight='bold', color='darkblue')
ax.set_title("Navigation Impairment (Non-Immersive vs Virtual Reality)", fontsize=18, weight='bold', color='darkblue')

# Save the figure
file_name = f"{Output_Dir}CSE/Normalized_CSE_2DyRV_B.png"
plt.savefig(file_name)
plt.clf()
print(f"--Completo Grafico de Resumen")

print('End of File')