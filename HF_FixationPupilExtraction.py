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
# -----------------------------------------------------------------------
#%%

import HA_ModuloArchivos as H_Mod
import pyxdf
import pandas as pd
import numpy as np

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# EJEMPLO DE USO de Explorar_DF .... con esto obtuvimos la descripción de los archivos .csv
#Explorar_Dir = Sujetos_Dir + "P02/PUPIL_LAB/P02/000/exports/000/"
#dataframes = H_Mod.Explorar_DF(Explorar_Dir)

Dir = Sujetos_Dir + "P33/"
Reporte, file = H_Mod.Grab_LabRecorderFile("NI", Dir)
print(Reporte)
file_path = Sujetos_Dir + "P33/PUPIL_LAB/000/exports/000/" + "surfaces/fixations_on_surface_Hefestp 1.csv"
Interesting_df = pd.read_csv(file_path)


for f in file:
    data, header = pyxdf.load_xdf(f)
        # data es una lista de STREAMS, que me desayuno, no parecen estar siempre en el mismo orden. Mejor chequear el nombre
        # los data['info']['name'][0] son 'Overwatch-Markers'. 'Overwatch-Joy', 'Overwatch-VR', 'pupil_capture'
    for d in data:
        if (d['info']['name'][0] == 'Overwatch-Markers') and (len(d['time_stamps']>20)):
            time_stamp = d['time_stamps']
            MarkersAlfa = H_Mod.Extract(d['time_series'], 0)  # Stream OverWatch Markers, Canal 0: Marker Primario
            MarkersBeta = H_Mod.Extract(d['time_series'], 1)  # Stream OverWatch Markerse, Canal 1: Marker Secundario
            Markers_df = pd.DataFrame(list(zip(time_stamp, MarkersAlfa, MarkersBeta)),
                                      columns=['OverWatch_time_stamp', 'OverWatch_MarkerA', 'OverWatch_MarkerB'])
            MarkersA_df = Markers_df.loc[Markers_df['OverWatch_MarkerA'] != 'NONE']

            # Un segmento solo para chequear que tan sincronico es el reporte de eventos de LSL con MATLAB EEG
            # ------------------------------------------------------------------------------------------------
            export_for_MATLAB = MarkersA_df.copy()

            valid_labels = ["1", "10", "20", "30"]
            MarkersA_df_filtered = export_for_MATLAB.loc[MarkersA_df['OverWatch_MarkerA'].astype(str).isin(valid_labels)]

            # Renombrar las columnas
            export_for_MATLAB = MarkersA_df_filtered[['OverWatch_time_stamp', 'OverWatch_MarkerA']].rename(
                columns={
                    'OverWatch_time_stamp': 'latencyP',
                    'OverWatch_MarkerA': 'labelP'
                }
            )

            # Exportar a CSV
            archivo = Py_Processing_Dir + 'export_for_MATLAB.csv'
            export_for_MATLAB.to_csv(archivo, index=False)
            # ------------------------------------------------------------------------------------------------

            #Ahora que hemos extraido de LabRecorder una infinidad de OverWatch Markers, tenemos que limpiar esta lista, dado
            #que está llena de errores y quedarnos solo con 66 marcadores de inicio y final para cada Trial y sus timestamps
            e, df = H_Mod.ClearMarkers(MarkersA_df) # e es un indicador de exito, df es la DataFrame con los indicadores de inicio, final, trial y timestamp
            e2, OverWatch_ClearedMarkers_df_Legacy = H_Mod.ClearMarkers_LEGACY(MarkersA_df)  # Aqui la versión LEGACY es la antigua que hice exitosa, pero indecifrable. La nueva es más legible y espero igual de exitosa

            print(f"Porcentaje de éxito: {e:.2f}%")
            print(f"Porcentaje de éxito: {e2:.2f}%")
            df = df.reset_index(drop=True)
            #  Esta DF viene con 66 lineas, cada una con una timestamp, cada una con un Trial number, la mitad con un START, la mitad con un STOP.

            # Limpiamos errores que tengan más de un STAR STOP por trial.
            df = H_Mod.LimpiarErroresdeOverwatch1(df)

            # Transformamos la DF por timestamp en una basada en Trials (con una columna para Timestamps de Start y otra para TimeStamps de Stop
            # añadimo la generación de trial_labels, preparandonos para hacer bins.
            trial_labels, trials = H_Mod.Markers_by_trial(df)

            # Ahora, en base a la columna de tiempo elegida (Timestamp); genera una columna OW_Trial que designa que Trial estaba ocurriendo en cada momento.
            Interesting_df = H_Mod.Binnear_DF(trial_labels, trials, Interesting_df, 'start_timestamp')

file_path = Py_Processing_Dir + "P33_eventos_exportados_desdeMATLAB.csv"
MAT_df = pd.read_csv(file_path)

# 1. Mantener solo las primeras 4 filas de `MAT_df`
MAT_df = MAT_df.iloc[:4].reset_index(drop=True)
# 2. Renombrar columnas en `MAT_df` para diferenciarlas si no están ya diferenciadas
MAT_df = MAT_df.rename(columns={'Label': 'labelMAT', 'Latency': 'latencyMAT'})
MAT_df['latencyMAT'] = MAT_df['latencyMAT'] / 1000

export_for_MATLAB = export_for_MATLAB.reset_index(drop=True)
# 3. Unir `MAT_df` y `export_for_MATLAB` (que tiene columnas labelP y latencyP) lado a lado
combined_df = pd.concat([MAT_df, export_for_MATLAB], axis=1)

# 4. Crear una nueva columna con la diferencia de latencias
combined_df['latency_diff'] = (combined_df['latencyP'] - combined_df['latencyMAT'])

# 5. Calcular el promedio y la dispersión (desviación estándar) de las diferencias
latency_mean = combined_df['latency_diff'].mean()
latency_std = combined_df['latency_diff'].std()

# 6. Mostrar resultados
print("Diferencias de latencias:")
print(combined_df)

print("\nPromedio de diferencias:", latency_mean)
print("Desviación estándar de diferencias:", latency_std)

max_diff = combined_df['latency_diff'].max() - combined_df['latency_diff'].min()

# Mostrar el resultado
print("Máxima diferencia de latencias  (ms):", max_diff*1000)

# Calcular diferencia entre el mínimo y el máximo de latencyMAT
latencyMAT_range = MAT_df['latencyMAT'].max() - MAT_df['latencyMAT'].min()

# Calcular diferencia entre el mínimo y el máximo de latencyP
latencyP_range = export_for_MATLAB['latencyP'].max() - export_for_MATLAB['latencyP'].min()

# Mostrar los resultados
print("Rango de latencias en MAT_df (latencyMAT) (s):", latencyMAT_range)
print("Rango de latencias en export_for_MATLAB (latencyP) (s) :", latencyP_range)
