#-------------------------------------------
#
#   Mayo 13, 2024.
#   Vamos a revisar los headKinematics.
#
#-------------------------------------------
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

df = pd.read_csv(Py_Processing_Dir+'CB_HeadKinematics.csv')
filtered_df = df[
    (df['Modality'] == 'Realidad Virtual') &
    (df['MWM_Bloque'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']))
]

# Agrupar por 'Subject' y calcular el promedio de las desviaciones estándar de las variables de interés
summary_df = filtered_df.groupby('Subject').agg({
    'vX_std': 'mean',
    'vY_std': 'mean',
    'vZ_std': 'mean',
    'vRoll_std': 'mean',
    'vJaw_std': 'mean',
    'vPitch_std': 'mean',
    'Grupo': 'first'  # Asumiendo que todos los registros por subject tienen el mismo grupo
}).reset_index()

print(summary_df)