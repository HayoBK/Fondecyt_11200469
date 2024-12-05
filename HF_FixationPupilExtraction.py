# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Diciembre - 05, Miércoles.
# Script para procesar de Pupil Labs usando Lab Recorder como indice
# Pero incorporando datos de Fijaciones
# -----------------------------------------------------------------------
#
# -----------------------------------------------------------------------
#%%
import pandas as pd
import glob2
import os
import pyxdf
import numpy as np
import HA_ModuloArchivos as H_Mod

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

Explorar_Dir = Sujetos_Dir + "P02/PUPIL_LAB/P02/000/exports/000/"

dataframes = {}

# Buscar archivos .csv en el directorio y subdirectorios
for subdir, _, files in os.walk(Explorar_Dir):
    for file in files:
        if file.endswith(".csv"):
            # Crear el nombre del DataFrame basado en la ruta relativa y el nombre del archivo
            relative_path = os.path.relpath(subdir, Explorar_Dir).replace(os.sep, "_")
            csv_name = file.replace(".csv", "")
            df_name = f"{relative_path}_{csv_name}".strip("_")

            # Leer el archivo .csv y almacenar el DataFrame
            file_path = os.path.join(subdir, file)
            dataframes[df_name] = pd.read_csv(file_path)
            print(f"DataFrame creado: {df_name}")
