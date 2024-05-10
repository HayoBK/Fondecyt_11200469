#%% ------------------------------------------------
#
#
# Veamos si podemos extra
#
#
#
# -------------------------------------------------

import pandas as pd
import glob2
import os
import pyxdf
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
import numpy as np
from pathlib import Path
import socket

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

nombre_host = socket.gethostname()
print(nombre_host)
if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

Subject_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
#%%

#Traer Los Datos de Navegacion... no se si serán tan importantes en realidad. para este.
NaviCSE_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
df = NaviCSE_df.copy()

#Este pedacito de codigo permite añadir OverWatch Trial como código a la DF de Simian.
def MWM_to_OW_trials (df):
    OW_t = 100
    New_column = []
    for row in df.itertuples():
        if row.True_Block == 'FreeNav':
            OW_t = 1
        elif row.True_Block == 'Training':
            OW_t = 2
        elif row.True_Block == 'VisibleTarget_1':
            OW_t = 2 + row.True_Trial
        elif row.True_Block == 'VisibleTarget_2':
            OW_t = 30 + row.True_Trial
        elif row.True_Block == 'HiddenTarget_1':
            OW_t = 6 + row.True_Trial
        elif row.True_Block == 'HiddenTarget_2':
            OW_t = 14 + row.True_Trial
        elif row.True_Block == 'HiddenTarget_3':
            OW_t = 22 + row.True_Trial
        New_column.append(OW_t)
    df['OW_trial'] = New_column
    return df

df = MWM_to_OW_trials(df)


#%% --------------Adquiramos los datos de Lab Recorder ------------------------------------

subject_folders = [f for f in os.listdir(Subject_Dir) if
                   os.path.isdir(os.path.join(Subject_Dir, f)) and f.startswith('P')]

# Diccionario para almacenar los datos de cada sujeto
subjects_data = {}

def explore_xdf_file(xdf_path):
    """Load and explore the structure of an XDF file."""
    data, header = pyxdf.load_xdf(xdf_path)

    print("Header Information:")
    print(header)

    print("\nStream Information:")
    for stream in data:
        print(f"Stream ID: {stream['info']['stream_id']} - {stream['info']['name'][0]}")
        print(f"Stream Type: {stream['info']['type'][0]}")
        print(f"Channel Count: {stream['info']['channel_count'][0]}")
        print(f"Nominal Sampling Rate: {stream['info']['nominal_srate'][0]}")
        print(f"Channel Format: {stream['info']['channel_format'][0]}")
        print(f"Stream Source ID: {stream['info']['source_id'][0]}")
        print("First few data points:", stream['time_series'][:5])
        print("Corresponding timestamps:", stream['time_stamps'][:5], "\n")



for subject_folder in subject_folders:
    # Construye el directorio para el sujeto actual
    dir_path = os.path.join(Subject_Dir, subject_folder, "LSL_LAB")
    # Patrón para buscar archivos .xdf
    pattern = os.path.join(dir_path, "**/*NI*.xdf")
    # Encuentra todos los archivos que coincidan con el patrón
    xdf_files = glob2.glob(pattern, recursive=True)

    for xdf_file in xdf_files:
        print(f"Processing file: {xdf_file} for subject: {subject_folder}")
        if subject_folder == 'P06':
            explore_xdf_file(xdf_file)

    if xdf_files:
        # Inicializa una lista para guardar los datos del sujeto
        subjects_data[subject_folder] = []
        for xdf_file in xdf_files:
            # Carga los datos del archivo .xdf
            data, header = pyxdf.load_xdf(xdf_file)
            # Guarda los datos y el nombre del archivo en la lista del sujeto
            subjects_data[subject_folder].append((xdf_file, data, header))
            # Imprime el nombre del archivo para verificar
            print(f"Archivo cargado para {subject_folder}: {xdf_file}")
    else:
        print(f"No se encontraron archivos .xdf para el sujeto {subject_folder}")




