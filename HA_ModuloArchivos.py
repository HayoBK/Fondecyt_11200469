#---------------------------------------------
# Por fin un modulo para tener la cosa más ordenada
# Para no tener que cambiar en cada nuevo script para cada computador
# distinto la ruta
#---------------------------------------------
# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Diciembre - 05, Miércoles.
# Script para procesar de Pupil Labs usando Lab Recorder como indice
# Pero para Modalidad RV
# -----------------------------------------------------------------------
# def Nombrar_HomePath(mi_path) --> Para definir ruta en cada compu
# def Explorar_DF(Dir):  --> Para explorar .csv en un directorio y revisarlos usando "view as DataFrame"
# def Grab_LabRecorderFile(Modalidad,mi_path): --> para capturar los archivos de LabRecorder en LUCIEN
# def ClearMarkers(MarkersA_df):  --> Codigo para extraer de LabRecorder Time_stamps asociados a START STOP de cada Trial
# def ClearMarkers_LEGACY(MarkersA_df):  --> Version original, efectiva, pero bien sucia
# def Extract(lst,place): --> Extrae un elemento de una lista
# def LimpiarErroresdeOverwatch1(df)  --> Limpiamos errores de trials que tengan más de un Start-stop
# def Markers_by_trial(df):  --> Toma la lista de OverwatchMarkers armada por timestamp y la ordena por Trial (lista para Bins)
# def Binnear_DF(trial_labels, trials, df, TimeMarker): En base a la columna "TimeMarker" con timestamp, y usando los bins de trial_label y trials, señala en que
#                                                       trial esta cada fila. (binnea)
# def Exportar_a_MATLAB_Sync(df,Sujetos_Dir,Sujeto): Para exportar datos del LSL Timestamps basados en los trial iniciales para marcar el delta de relojes de MATLAB y LSL

# -----------------------------------------------------------------------

# Lexico para el Flujo de Datos a MATLAB:
# P_LEFT = 4
# P_RIGHT = 6
# P_FORWARD = 8
# P_BACK = 2
# P_STILL = 5
# P_TRial es igual a 100 + Numero de Trial
# P_FULLSTOP = 202
# P_POSSIBLE_STOP = 201
# P_FALSE_STOP = 203
# P_GO_ON = 200
# P_FORCE_START = 205


import socket
import glob2
import pyxdf
from io import StringIO
import sys
import os
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------


def Nombrar_HomePath(mi_path):
    # Un ejemplo de la ruta luego de La ruta base "002-LUCIEN/Py_INFINITE/df_PsicoCognitivo/"
    # Busca en que compu estamos
    # y luego genera la ruta en One-Drive hasta la carpeta del Fondecyt

    print('Proceso de HA_ModuloArchivos, Identificamos en que computador estamos... ')
    nombre_host = socket.gethostname()
    print('Estamos en: ',nombre_host)

    if nombre_host == 'DESKTOP-PQ9KP6K':
        home = "D:/Mumin_UCh_OneDrive"

    if nombre_host == 'MSI':
        home = "D:/Titan-OneDrive"

    if nombre_host == 'iMac-de-Hayo.local':
        home = "/Users/hayo/Library/CloudStorage/OneDrive-Personal"

    ruta = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/" + mi_path
    if nombre_host == 'iMac-de-Hayo.local':
        home = "/Users/hayo/Library/CloudStorage/OneDrive-Personal"
        ruta = home + "/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/" + mi_path
    print('Definimos una Ruta dirigiendonos a: ',ruta)
    print('--------------- CHECK -------------')
    print(' ')
    return ruta

# -----------------------------------------------------------------------


def Explorar_DF(Dir):
    # Buscar archivos .csv en el directorio y subdirectorios, para poder explorarlos.

    dataframes = {}
    for subdir, _, files in os.walk(Dir):
        for file in files:
            if file.endswith(".csv"):
                # Crear el nombre del DataFrame basado en la ruta relativa y el nombre del archivo
                relative_path = os.path.relpath(subdir, Dir).replace(os.sep, "_")
                csv_name = file.replace(".csv", "")
                df_name = f"{relative_path}_{csv_name}".strip("_")

                # Leer el archivo .csv y almacenar el DataFrame
                file_path = os.path.join(subdir, file)
                dataframes[df_name] = pd.read_csv(file_path)
                print(f"DataFrame creado: {df_name}")
    return dataframes
# EJEMPLO DE USO.
#Explorar_Dir = Sujetos_Dir + "P02/PUPIL_LAB/P02/000/exports/000/"
#dataframes = Explorar_DF(Explorar_Dir)

# -----------------------------------------------------------------------


def Grab_LabRecorderFile(Modalidad,mi_path):
    # Modalidad es NI o RV
    # mi_path Ruta que marque a el Directorio Base de cada Paciente PXX
    # devuelve: exito (reporte de como nos fue) y XDF_Files: una lista de los archivos de LabRecorder Validos
    exito= ["No se capturó ningun archivo"]

    if Modalidad == 'NI':
        patrones = [
            "/LSL_LAB/**/*NI*.xdf",  # Patrón existente
            "/LSL_LAB/**/*No Inmer*.xdf",  # Ejemplo de un nuevo patrón
            "/LSL_LAB/**/*Blufagondi*.xdf"  # Otro patrón adicional
        ]
    if Modalidad == 'RV':
        patrones = [
            "/LSL_LAB/**/*RV*.xdf",  # Patrón existente
            "/LSL_LAB/**/*Virt*.xdf",  # Ejemplo de un nuevo patrón
            "/LSL_LAB/**/*Real*.xdf"  # Otro patrón adicional
        ]
    XDF_files = []
    XDF_files_validated =[]
    for p in patrones:
        px = mi_path + p
        XDF_files += glob2.glob(px)

    if XDF_files:
        exito=[]
        for x in XDF_files:
            try:
                # Redirigir stderr para capturar mensajes de pyxdf
                stderr_capture = StringIO()
                sys.stderr = stderr_capture  # Redirige stderr a una variable

                data, header = pyxdf.load_xdf(x)  # Intentar cargar el archivo

                sys.stderr = sys.__stderr__  # Restaurar stderr

                # Verificar si hubo mensajes de corrupción
                stderr_output = stderr_capture.getvalue()
                if "likely XDF file corruption" in stderr_output:
                    e = str(stderr_capture) + ' en ' + mi_path
                    exito.append(e)
                    raise ValueError(f"Archivo corrupto detectado por stderr: {stderr_output.strip()}")

                XDF_files_validated.append(x)  # Si no hay problemas, el archivo es válido
                #print(f"Validado: {x}")

            except ValueError as e:
                # Manejar archivos corruptos
                print(f"Archivo corrupto detectado: {x}. Error: {e}")


            finally:
                sys.stderr = sys.__stderr__  # Restaurar stderr incluso si ocurre un error


    # Reemplazar XDF_files con la lista de archivos validados
    XDF_files = XDF_files_validated

    if XDF_files:
        l = len(XDF_files)
        exito = "Se capturaron " + str(l) +" archivo(s) válido"

    return exito, XDF_files

# -----------------------------------------------------------------------
def Extract(lst,place):
    return [item[place] for item in lst]
# -----------------------------------------------------------------------

def ClearMarkers(MarkersA_df):
    # Listas para almacenar los resultados procesados
    TimePoint = []   # Almacena los tipos de evento ("START" o "STOP")
    TimeStamp2 = []  # Almacena las marcas de tiempo asociadas a los eventos
    Trial = []       # Almacena los identificadores de cada trial

    # Variables de estado
    LastTrial = 0          # Último trial procesado
    LastTrial_Length = 0   # Duración del último trial registrado
    t1, t2 = 0, 0          # Tiempos de inicio y fin del trial activo
    OnGoing = False        # Indica si hay un trial en curso
    started = False        # Indica si ya se ha iniciado algún trial
    confirmedSTOP = False  # Indica si un STOP ha sido confirmado
    Ts = 0                 # Marca de tiempo actual


    # Iteración sobre cada fila del DataFrame de entrada
    for row in MarkersA_df.itertuples():
        TP = 'NONE'  # Inicializa el tipo de evento como "NONE" (sin evento válido)
        Tr = 1000    # Inicializa el identificador del trial con un valor de error
        Ts = row.OverWatch_time_stamp  # Obtiene la marca de tiempo de la fila actual

        # Caso 1: Detectar un evento de inicio (START)
        if row.OverWatch_MarkerA.isdigit() and int(row.OverWatch_MarkerA) < 34:
            started = True                 # Marca que se ha iniciado el procesamiento
            TP = 'START'                  # Etiqueta el evento como "START"
            Tr = int(row.OverWatch_MarkerA)  # Identificador del trial
            OnGoing = True                # Indica que el trial está activo
            confirmedSTOP = False         # Reinicia el estado de STOP confirmado

            # Si el trial actual ya había sido registrado, elimina duplicados
            if Tr == LastTrial:
                TimeStamp2 = TimeStamp2[:-2]
                TimePoint = TimePoint[:-2]
                Trial = Trial[:-2]

            # Actualiza el último trial procesado
            LastTrial = Tr
            t1 = Ts  # Registra el tiempo de inicio del trial

        # Caso 2: Corregir un "Falso Stop"
        if row.OverWatch_MarkerA == 'Falso Stop' and started:
            # Si no hay un trial en curso pero hay registros previos
            if not OnGoing and len(Trial) > 0:
                OnGoing = True  # Marca que ahora hay un trial en curso
                del TimeStamp2[-1]  # Elimina el último evento registrado
                del TimePoint[-1]
                del Trial[-1]

        # Caso 3: Detectar un evento de finalización (STOP)
        if row.OverWatch_MarkerA == 'Stop' and started:
            if OnGoing:
                # Si hay un trial activo, registra el STOP
                TP = 'STOP'
                Tr = LastTrial
                OnGoing = False  # Marca que el trial ha finalizado
                t2 = Ts          # Registra el tiempo de finalización
                LastTrial_Length = t2 - t1  # Calcula la duración del trial
            else:
                # Si no hay un trial activo, maneja un STOP inesperado
                if LastTrial < 32:
                    if (Ts - LastTrial_Length) > t2:
                        # Calcula un tiempo ficticio para un "START" previo
                        TimeStamp2.append(Ts - LastTrial_Length)
                    else:
                        # Calcula un tiempo ficticio basado en el 90% del lapso
                        Lapse90 = (Ts - t2) * 0.9
                        TimeStamp2.append(Ts - Lapse90)

                    confirmedSTOP = False  # Marca que el STOP no está confirmado
                    TimePoint.append('START')  # Agrega un "START" ficticio
                    LastTrial += 1  # Incrementa el identificador del trial
                    Trial.append(LastTrial)
                    TP = 'STOP'     # Registra el evento actual como "STOP"
                    Tr = LastTrial
                    OnGoing = False
                    t2 = Ts         # Actualiza el tiempo de finalización ficticio
                    LastTrial_Length = t2 - t1

        # Caso 4: Detectar un "STOP confirmado"
        if row.OverWatch_MarkerA == 'Stop confirmado' and started:
            if OnGoing:
                # Si hay un trial activo, registra un STOP confirmado
                TP = 'STOP'
                Tr = LastTrial
                OnGoing = False
                confirmedSTOP = True

        # Imprime los resultados parciales para seguimiento

        # Si se detectó un evento válido, almacena los resultados
        if TP != 'NONE':
            TimeStamp2.append(Ts)
            TimePoint.append(TP)
            Trial.append(Tr)

    # Construcción del DataFrame de salida
    output = pd.DataFrame(list(zip(TimeStamp2, TimePoint, Trial)),
                          columns=['OverWatch_time_stamp', 'OverWatch_MainMarker', 'OverWatch_Trial'])
    # Filtra los eventos "NONE" que no son válidos
    output = output.loc[output['OverWatch_MainMarker'] != 'NONE']

    # Cálculo del porcentaje de éxito
    num_rows = len(output)  # Número de filas en el DataFrame de salida
    exito = (num_rows / 66) * 100  # Porcentaje de éxito basado en 66 filas esperadas

    return exito, output



def ClearMarkers_LEGACY(MarkersA_df):

    TimePoint = []
    TimeStamp2 = []
    Trial = []
    LastTrial = 0
    LastTrial_Length = 0
    t1=0
    t2=0
    OnGoing = False
    started = False
    confirmedSTOP = False
    Ts = 0
    #print('-----------------------')
    #print('Inicio primera revisión')
    #print('-----------------------')

    for row in MarkersA_df.itertuples():
        TP = 'NONE' # Para alimentar la lista de TimePoints
        Tr = 1000 # Para alimentar la lista de trials. Marcando un error
        Ts = row.OverWatch_time_stamp
        if (row.OverWatch_MarkerA.isdigit()) and (int(row.OverWatch_MarkerA) < 34) :
            started = True
            TP = 'START'
            Tr = int(row.OverWatch_MarkerA)
            OnGoing = True
            confirmedSTOP = False

            if Tr == LastTrial:
                #print('Borrados ', TimeStamp2[-2], TimeStamp2[-1])
                TimeStamp2 = TimeStamp2[:len(TimeStamp2) - 2]

                TimePoint = TimePoint[:len(TimePoint) - 2]
                Trial = Trial[:len(Trial) - 2]
            LastTrial = Tr

            t1=Ts
        if (row.OverWatch_MarkerA == 'Falso Stop') and (started==True):
            if (OnGoing == False) and (len(Trial)>0):
                OnGoing = True
                del TimeStamp2[-1]
                del TimePoint[-1]
                del Trial[-1]

        if (row.OverWatch_MarkerA == 'Stop') and (started == True):
            if OnGoing:
                TP = 'STOP'
                Tr = LastTrial
                OnGoing = False
                t2=Ts
                LastTrial_Length=t2-t1
            else:
                # Esta es la situación donde se supone que NO hay un Trial activo, pero
                # Se encuentra una señal de stop... implica que hubo un trial
                # no correctamente inicializado
                if (LastTrial < 32):
                    if (Ts-LastTrial_Length) > t2:
                        TimeStamp2.append(Ts-LastTrial_Length)
                    else:
                        Lapse90=(Ts-t2)*0.9
                        TimeStamp2.append(Ts-Lapse90)
                    confirmedSTOP = False
                    TimePoint.append('START')
                    LastTrial+=1
                    Trial.append(LastTrial)
                    TP = 'STOP'
                    Tr = LastTrial
                    OnGoing = False
                    t2 = Ts
                    LastTrial_Length = t2 - t1


        if (row.OverWatch_MarkerA == 'Stop confirmado') and (started == True):
            if OnGoing: # Es caso de que no haya registro de un Stop previo
                TP = 'STOP'
                Tr = LastTrial
                OnGoing = False
                confirmedSTOP = True

        #print(Ts,TP,Tr)
        if TP != 'NONE':
            TimeStamp2.append(Ts)
            TimePoint.append(TP)
            Trial.append(Tr)

    output = pd.DataFrame(list(zip(TimeStamp2, TimePoint, Trial)),
                          columns =['OverWatch_time_stamp', 'OverWatch_MainMarker', 'OverWatch_Trial'])
    output = output.loc[output['OverWatch_MainMarker'] != 'NONE']
    num_rows = len(output)  # Número de filas en el DataFrame de salida
    exito = (num_rows / 66) * 100  # Porcentaje de éxito basado en 66 filas esperadas

    return exito, output


def LimpiarErroresdeOverwatch1(df):
    if not df.empty:
        dfC = df
        # Contar eventos por cada Trial
        event_counts = dfC.groupby(['OverWatch_Trial', 'OverWatch_MainMarker']).size().unstack(fill_value=0)
        # Filtrar trials que tienen exactamente un START y un STOP
        valid_trials = event_counts[(event_counts['START'] == 1) & (event_counts['STOP'] == 1)].index
        # Filtrar el DataFrame original para mantener solo los trials válidos
        df = dfC[dfC['OverWatch_Trial'].isin(valid_trials)]
        # Mostrar el DataFrame limpio
    return df


def Markers_by_trial(df):
    inicios = df[df['OverWatch_MainMarker'] == 'START']['OverWatch_time_stamp'].reset_index(drop=True)
    OW_Trials = df[df['OverWatch_MainMarker'] == 'START']['OverWatch_Trial'].reset_index(drop=True)
    finales = df[df['OverWatch_MainMarker'] == 'STOP']['OverWatch_time_stamp'].reset_index(drop=True)
    trials = pd.DataFrame({'Start': inicios, 'End': finales, 'OW_trials': OW_Trials})

    trial_labels = list(range(1, 34))
    trial_labels = trials['OW_trials'].tolist()  # Ojo aqui que quede bien...
    trial_labels = np.array(trial_labels)
    return trial_labels, trials


def Binnear_DF(trial_labels, trials, df, TimeMarker):
    bins = pd.IntervalIndex.from_tuples(list(zip(trials['Start'], trials['End'])), closed='left')
    try_df = df
    try_df['OW_Trial'] = pd.cut(df[TimeMarker], bins).map(dict(zip(bins, trial_labels)))
    return try_df

def Exportar_a_MATLAB_Sync(df):
    export_for_MATLAB = df.copy()

    valid_labels = ["1", "10", "20", "30"]
    MarkersA_df_filtered = export_for_MATLAB.loc[df['OverWatch_MarkerA'].astype(str).isin(valid_labels)]
    # Renombrar las columnas
    export_for_MATLAB = MarkersA_df_filtered[['OverWatch_time_stamp', 'OverWatch_MarkerA']].rename(
        columns={
            'OverWatch_time_stamp': 'latencyLSL',
            'OverWatch_MarkerA': 'labelLSL'
        }
    )
    # Exportar a CSV --> Update --> mejor sacar la exportación ed
    #export_for_MATLAB = export_for_MATLAB.reset_index()
    #archivo = Sujetos_Dir + Sujeto + "/EEG/" + 'export_for_MATLAB_Sync.csv'
    #export_for_MATLAB.to_csv(archivo, index=False)
    return export_for_MATLAB



def process_xdf_files(files):
    """
    Procesa una lista de archivos XDF y extrae información relevante.

    Parámetros:
    files (list): Lista de rutas de archivos XDF a procesar.

    Retorna:
    tuple: Contiene las siguientes variables:
        - processed_files (int): Número de archivos procesados exitosamente.
        - sync_df (DataFrame): DataFrame para sincronización de eventos.
        - trials_per_timestamp (DataFrame): DataFrame con información de trials por timestamp.
        - trials_per_trialLabel (DataFrame): DataFrame con información de trials por etiqueta.
        - trial_labels (list): Lista de etiquetas de trials.
    """
    processed_files = 0
    sync_df = pd.DataFrame()
    trials_per_timestamp = pd.DataFrame()
    trials_per_trialLabel = pd.DataFrame()
    trial_labels = []

    for f in files:
        data, header = pyxdf.load_xdf(f)
        for d in data:
            if (d['info']['name'][0] == 'Overwatch-Markers') and (len(d['time_stamps']) > 20):
                processed_files += 1
                time_stamp = d['time_stamps']
                markers_alfa = Extract(d['time_series'], 0)
                markers_beta = Extract(d['time_series'], 1)
                markers_df = pd.DataFrame({
                    'OverWatch_time_stamp': time_stamp,
                    'OverWatch_MarkerA': markers_alfa,
                    'OverWatch_MarkerB': markers_beta
                })
                markers_a_df = markers_df[markers_df['OverWatch_MarkerA'] != 'NONE']

                # Generar archivo para sincronización
                sync_df = Exportar_a_MATLAB_Sync(markers_a_df)
                sync_df = sync_df.reset_index()
                # Limpiar y procesar marcadores
                success_rate, df = ClearMarkers(markers_a_df)
                legacy_success_rate, legacy_df = ClearMarkers_LEGACY(markers_a_df)
                print("Comprobación de éxito en la extracción de marcadores de LSL")
                print(f"Porcentaje de éxito (versión nueva): {success_rate:.2f}%")
                print(f"Porcentaje de éxito (versión LEGACY): {legacy_success_rate:.2f}%")
                df = df.reset_index(drop=True)

                # Limpiar errores y estructurar datos por trial
                df = LimpiarErroresdeOverwatch1(df)
                trial_labels, trials_per_trialLabel = Markers_by_trial(df)
                trials_per_timestamp = df

    return processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels


def enriquecer_fijaciones_firstfix(fixfile_path, trialfile_path, modalidad):
    # Base de etiquetas
    MWM_labels_base = [
        'T01_FreeNav', 'T02_Training',
        'T03_NaviVT1_i1', 'T04_NaviVT1_i2', 'T05_NaviVT1_i3', 'T06_NaviVT1_i4',
        'T07_NaviHT1_i1', 'T08_NaviHT1_i2', 'T09_NaviHT1_i3', 'T10_NaviHT1_i4',
        'T11_NaviHT1_i5', 'T12_NaviHT1_i6', 'T13_NaviHT1_i7', 'T14_Rest1',
        'T15_NaviHT2_i1', 'T16_NaviHT2_i2', 'T17_NaviHT2_i3', 'T18_NaviVT2_i4',
        'T19_NaviHT2_i5', 'T20_NaviHT2_i6', 'T21_NaviHT2_i7', 'T22_Rest2',
        'T23_NaviHT3_i1', 'T24_NaviHT3_i2', 'T25_NaviHT3_i3', 'T26_NaviHT3_i4',
        'T27_NaviHT3_i5', 'T28_NaviHT3_i6', 'T29_NaviHT3_i7', 'T30_Rest3',
        'T31_NaviVT2_i1', 'T32_NaviVT2_i2', 'T33_NaviVT2_i3'
    ]
    MWM_labels = [f"{modalidad}_{label}" for label in MWM_labels_base]
    label_map = {i + 1: MWM_labels[i] for i in range(len(MWM_labels))}

    # Cargar archivos
    if not os.path.exists(fixfile_path) or not os.path.exists(trialfile_path):
        print(f"⚠️ No se encontró el archivo de fijaciones o de trials.")
        return

    df = pd.read_csv(fixfile_path)
    trials = pd.read_csv(trialfile_path)

    # Agregar columna Real_Trial (texto enriquecido)
    if 'Real_Trial' not in df.columns:
        df['trial_id'] = np.nan
        for _, row in trials.iterrows():
            mask = (df['start_time'] >= row['start_time']) & (df['start_time'] <= row['end_time'])
            df.loc[mask, 'trial_id'] = row['trial_id']
        df['trial_id'] = df['trial_id'].astype('Int64')  # permitir NA
        df['Real_Trial'] = df['trial_id'].map(label_map)

    if 'is_first_fixation' in df.columns:
        print("✅ Ya contiene 'is_first_fixation'. No se recalcula.")
    else:
        print("🛠️ Calculando primeras fijaciones...")

        if not all(col in df.columns for col in ['start_time', 'duration', 'eye_x', 'eye_y']):
            print("❌ Faltan columnas necesarias.")
            return

        df['on_screen'] = df['eye_x'].between(0, 1) & df['eye_y'].between(0, 1)
        df = df.sort_values(by='start_time').reset_index(drop=True)

        df['prev_x'] = df['eye_x'].shift(1)
        df['prev_y'] = df['eye_y'].shift(1)
        df['dist'] = np.sqrt((df['eye_x'] - df['prev_x']) ** 2 + (df['eye_y'] - df['prev_y']) ** 2)

        df['is_first_fixation'] = df['dist'] > 0.2
        df.loc[0, 'is_first_fixation'] = True
        df['group_id'] = df['is_first_fixation'].cumsum()

    # Guardar archivo enriquecido
    enriched_path = fixfile_path.replace('.csv', '_enriched.csv')
    df.to_csv(enriched_path, index=False)
    print(f"✅ Guardado archivo enriquecido: {enriched_path}")

    # Agrupar por group_id y Real_Trial
    group_stats = df.groupby(['group_id', 'Real_Trial']).agg(
        first_start=('start_time', 'first'),
        n_fijaciones=('start_time', 'count'),
        duracion_total=('duration', 'sum'),
        std_x=('eye_x', 'std'),
        std_y=('eye_y', 'std'),
        x_inicio=('eye_x', 'first'),
        y_inicio=('eye_y', 'first')
    ).reset_index()

    resumen_path = fixfile_path.replace('.csv', '_group_summary.csv')
    group_stats.to_csv(resumen_path, index=False)
    print(f"📄 Guardado resumen de grupos: {resumen_path}")