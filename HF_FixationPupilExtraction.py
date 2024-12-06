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

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# EJEMPLO DE USO de Explorar_DF .... con esto obtuvimos la descripción de los archivos .csv
#Explorar_Dir = Sujetos_Dir + "P02/PUPIL_LAB/P02/000/exports/000/"
#dataframes = Explorar_DF(Explorar_Dir)

Dir = Sujetos_Dir + "P02/"
Reporte, file = H_Mod.Grab_LabRecorderFile("NI", Dir)
print(Reporte)

def Extract(lst,place):
    return [item[place] for item in lst]

for f in file:
    data, header = pyxdf.load_xdf(f)
        # data es una lista de STREAMS, que me desayuno, no parecen estar siempre en el mismo orden. Mejor chequear el nombre
        # los data['info']['name'][0] son 'Overwatch-Markers'. 'Overwatch-Joy', 'Overwatch-VR', 'pupil_capture'
    for d in data:
        if d['info']['name'][0] == 'Overwatch-Markers':
            time_stamp = d['time_stamps']
            MarkersAlfa = Extract(d['time_series'], 0)  # Stream OverWatch Markers, Canal 0: Marker Primario
            MarkersBeta = Extract(d['time_series'], 1)  # Stream OverWatch Markerse, Canal 1: Marker Secundario
            Markers_df = pd.DataFrame(list(zip(time_stamp, MarkersAlfa, MarkersBeta)),
                                      columns=['OverWatch_time_stamp', 'OverWatch_MarkerA', 'OverWatch_MarkerB'])
            MarkersA_df = Markers_df.loc[Markers_df['OverWatch_MarkerA'] != 'NONE']
            # -------------------------------------------------------------
            # Vamos a iniciar el análisis de los Markers de OverWatch para
            # quedar con una lista confiable de marcadores


