# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Diciembre - 05, Miércoles.
# Script para procesar de Pupil Labs usando Lab Recorder como indice
# Pero incorporando datos de Fijaciones
# -----------------------------------------------------------------------
#   Dentro de un directorio export en los directorios de PupilLabs (Hay que hacer el export con Pupil Labs Player
#   hay:
#   1) Un archivo "surfaces/fixation_on_surface_Hefestp 1.csv"
#       fixation_id : Correlativo para identificar fijaciones
#       start_timestamp: inicio Fijación
#       duration (supongo son milisegundos)
#       on_surf: True o False
#       norm_pos_x & norm_pos_y : con respecto a la superficie
#   2) Un archivo "pupil_positions.csv"
#       diameter & diamater_3d: diametro pupilar. Ojo que 3d tiene hartos "nan" en que no fue capturada
#       eye_id: Identificación del ojo 0/1
#       pupil_timestamp: momento
#   3) Un archivo "blinks.csv"
#       id : para identificar cada pestañeo
#       start_timestamp
#       duration
#       end_timestamp
# -----------------------------------------------------------------------
#%%

import HA_ModuloArchivos as H_Mod
import pyxdf
import pandas as pd
import numpy as np

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# ------------------ EXPLORACION --------------------------------------
# EJEMPLO DE USO de Explorar_DF .... con esto obtuvimos la descripción de los archivos .csv
# Explorar_Dir = Sujetos_Dir + "P33/PUPIL_LAB/000/exports/000/"
# dataframes = H_Mod.Explorar_DF(Explorar_Dir)
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Ahora vamos a buscar los archivos XDF
Sujeto = 'P33'
Dir = Sujetos_Dir + "P33/"
# Buscamos los archivos XDF en modalidad "NI" -------------------------
Reporte, file = H_Mod.Grab_LabRecorderFile("NI", Dir)
print(Reporte)

# Ahora vamos a buscar los archivos csv de Fixations y Blinks
file_path = Sujetos_Dir + "P33/PUPIL_LAB/000/exports/000/" + "surfaces/fixations_on_surface_Hefestp 1.csv"
fixations_df = pd.read_csv(file_path)
Interesting_df = pd.read_csv(file_path)
file_path = Sujetos_Dir + "P33/PUPIL_LAB/000/exports/000/" + "blinks.csv"
blinks_df = pd.read_csv(file_path)


# Con esta Función entregamos los XDF files y recuperamos todo lo que necesitamos para la sincronización

processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels = H_Mod.process_xdf_files(file)
print('Checking: Se procesaron una Cantidad de archivo XDF igual a (ojalá 1): ',processed_files)

#Exportamos el Archivo de
archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'export_for_MATLAB_Sync_NI.csv'
sync_df.to_csv(archivo, index=False)



# ---------------------------------------------------------------------------------------------------------------------------------------------
# Ahora, en base a la columna de tiempo elegida (Timestamp); genera una columna OW_Trial que designa que Trial estaba ocurriendo en cada momento.
Interesting_df = H_Mod.Binnear_DF(trial_labels, trials_per_trialLabel, Interesting_df, 'start_timestamp')


# ------------ Exportar los csv necesarios para analisis en MATLAB ------------------

Sujeto = 'P33'

# Primero para Blinks
#--------------------->>>
output_df = blinks_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'  # Trials need start and end times, not duration
        })
output_df = output_df[['start_time', 'duration']]  # Keep only relevant columns

archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'blinks_forMATLAB_NI.csv'
output_df.to_csv(archivo, index=False)


# Ahora para Fixations
#--------------------->>>
output_df = fixations_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'  # Trials need start and end times, not duration
        })
output_df = output_df.drop_duplicates(subset='fixation_id', keep='first')
output_df = output_df[['start_time', 'duration']]  # Keep only relevant columns

archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'fixation_forMATLAB_NI.csv'
output_df.to_csv(archivo, index=False)

# Ahora para los TRIALS !!!!
#--------------------->>>
Modalidad = 'NI'
output_df = trials_per_trialLabel.rename(columns={
            'OW_trials': 'trial_id',
            'Start': 'start_time',
            'End': 'end_time'  # Trials need start and end times, not duration
        })
archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'trials_forMATLAB_'+Modalidad+'.csv'
output_df.to_csv(archivo, index=False)

# ---------------------------------------------------------------------------------------

# Buscamos los archivos XDF en modalidad "RV" -------------------------

# ---------------------------------------------------------------------------------------

Reporte, file = H_Mod.Grab_LabRecorderFile("RV", Dir)
print(Reporte)
# Ahora vamos a buscar los archivos csv de Fixations y Blinks
file_path = Sujetos_Dir + "P33/PUPIL_LAB/001/exports/000/" + "fixations.csv"  # Para RV
fixations_df = pd.read_csv(file_path)
Interesting_df = pd.read_csv(file_path)
file_path = Sujetos_Dir + "P33/PUPIL_LAB/001/exports/000/" + "blinks.csv"
blinks_df = pd.read_csv(file_path)


# Con esta Función entregamos los XDF files y recuperamos todo lo que necesitamos para la sincronización

processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels = H_Mod.process_xdf_files(file)
print('Checking: Se procesaron una Cantidad de archivo XDF igual a (ojalá 1): ',processed_files)

#Exportamos el Archivo de
archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'export_for_MATLAB_Sync_RV.csv'
sync_df.to_csv(archivo, index=False)

# ---------------------------------------------------------------------------------------------------------------------------------------------
# Ahora, en base a la columna de tiempo elegida (Timestamp); genera una columna OW_Trial que designa que Trial estaba ocurriendo en cada momento.
Interesting_df = H_Mod.Binnear_DF(trial_labels, trials_per_trialLabel, Interesting_df, 'start_timestamp')

# ------------ Exportar los csv necesarios para analisis en MATLAB ------------------

Sujeto = 'P33'

# Primero para Blinks
#--------------------->>>
output_df = blinks_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'  # Trials need start and end times, not duration
        })
output_df = output_df[['start_time', 'duration']]  # Keep only relevant columns

archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'blinks_forMATLAB_RV.csv'
output_df.to_csv(archivo, index=False)


# Ahora para Fixations
#--------------------->>>
output_df = fixations_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'  # Trials need start and end times, not duration
        })
output_df = output_df.drop_duplicates(subset='id', keep='first')  # Cambio para RV
output_df = output_df[['start_time', 'duration']]  # Keep only relevant columns

archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'fixation_forMATLAB_RV.csv'
output_df.to_csv(archivo, index=False)

# Ahora para los TRIALS !!!!
#--------------------->>>
Modalidad = 'RV'
output_df = trials_per_trialLabel.rename(columns={
            'OW_trials': 'trial_id',
            'Start': 'start_time',
            'End': 'end_time'  # Trials need start and end times, not duration
        })
archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'trials_forMATLAB_'+Modalidad+'.csv'
output_df.to_csv(archivo, index=False)


print(" Work's Done")