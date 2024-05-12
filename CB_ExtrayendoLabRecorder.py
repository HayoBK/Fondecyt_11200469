#%% ------------------------------------------------
#
#
# Veamos si podemos extra
#
# Datos de Overwatch-VR
# LSL_Stream2 = StreamInfo('Overwatch-VR','Datos VR', 6 , 0, 'float32','overwatch-Titan2')
# Canales = 7 : vx,vy,vz,vroll,vjaw,vpitch
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


# Diccionario para almacenar los datos de cada sujeto y modalidad
subjects_data = {}

def explore_xdf_file(xdf_path):
    """Función para explorar y obtener información básica de un archivo XDF."""
    data, header = pyxdf.load_xdf(xdf_path)
    print(f"Processing file: {xdf_path}")
    print("Header Information:", header)
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

def classify_and_store_xdf_files():
    subject_folders = [f for f in os.listdir(Subject_Dir) if os.path.isdir(os.path.join(Subject_Dir, f)) and f.startswith('P')]

    for subject_folder in subject_folders:
        dir_path = os.path.join(Subject_Dir, subject_folder)
        # Buscar recursivamente todos los archivos .xdf
        xdf_files = glob2.glob(os.path.join(dir_path, "**/*.xdf"), recursive=True)

        subjects_data[subject_folder] = {'Realidad Virtual': [], 'No Inmersivo': []}

        for xdf_file in xdf_files:
            if 'Realidad Virtual' in xdf_file or 'RV' in xdf_file or 'Realidad' in xdf_file or 'VR' in xdf_file:
                modality = 'Realidad Virtual'
            elif 'No Inmersivo' in xdf_file or 'NI' in xdf_file:
                modality = 'No Inmersivo'
            else:
                modality = 'No Inmersivo'  # Por defecto, si no se encuentra ninguna clave

            subjects_data[subject_folder][modality].append(xdf_file)
            #explore_xdf_file(xdf_file)
            print("Adquirido ", subject_folder, modality)


def report_xdf_files():
    # Ordenar los sujetos alfabéticamente
    sorted_subjects = sorted(subjects_data.keys())

    for subject in sorted_subjects:
        # Imprimir primero "No Inmersivo" y luego "Realidad Virtual"
        print(f"Sujeto {subject}, Modalidad No Inmersivo: {len(subjects_data[subject]['No Inmersivo'])} archivos")
        if subjects_data[subject]['No Inmersivo']:
            for file_path in subjects_data[subject]['No Inmersivo']:
                print(f"  - {os.path.basename(file_path)}")  # Imprime solo el nombre del archivo

        print(
            f"Sujeto {subject}, Modalidad Realidad Virtual: {len(subjects_data[subject]['Realidad Virtual'])} archivos")
        if subjects_data[subject]['Realidad Virtual']:
            for file_path in subjects_data[subject]['Realidad Virtual']:
                print(f"  - {os.path.basename(file_path)}")  # Imprime solo el nombre del archivo


# Ejecutar las funciones
classify_and_store_xdf_files()
report_xdf_files()
#explore_xdf_file(subjects_data['P06']['Realidad Virtual'][0])

#%%
def Extract(lst,place):
    return [item[place] for item in lst]
def ClearMarkers(MarkersA_df):

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
    print('-----------------------')
    print('Inicio primera revisión')
    print('-----------------------')

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
                print('Borrados ', TimeStamp2[-2], TimeStamp2[-1])
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

        print(Ts,TP,Tr)
        if TP != 'NONE':
            TimeStamp2.append(Ts)
            TimePoint.append(TP)
            Trial.append(Tr)

    output = pd.DataFrame(list(zip(TimeStamp2, TimePoint, Trial)),
                          columns =['OverWatch_time_stamp', 'OverWatch_MainMarker', 'OverWatch_Trial'])
    output = output.loc[output['OverWatch_MainMarker'] != 'NONE']
    return output
def extraerData_xdf_file(xdf_path):
    data, header = pyxdf.load_xdf(xdf_path)
    vr_data_df = pd.DataFrame()
    markers_df = pd.DataFrame()

    for d in data:
        if d['info']['name'][0] == 'Overwatch-Markers':
            time_stamp = d['time_stamps']
            MarkersAlfa = Extract(d['time_series'], 0)  # Stream OverWatch Markers, Canal 0: Marker Primario
            MarkersBeta = Extract(d['time_series'], 1)  # Stream pupilCapture, Canal 1: Marker Secundario
            Markers_df = pd.DataFrame(list(zip(time_stamp, MarkersAlfa, MarkersBeta)),
                                      columns=['OverWatch_time_stamp', 'OverWatch_MarkerA', 'OverWatch_MarkerB'])
            MarkersA_df = Markers_df.loc[Markers_df['OverWatch_MarkerA'] != 'NONE']
    # -------------------------------------------------------------
    # Vamos a iniciar el análisis de los Markers de OverWatch para
    # quedar con una lista confiable de marcadores
              # Pasar MarkersA_df aquí

            OverWatch_ClearedMarkers_df = ClearMarkers(MarkersA_df)  # Aqui construimos la base de datos con los marcadores con todas las
    # ... correcciones interpretativas identificadas hasta el momento.
            df = OverWatch_ClearedMarkers_df
            df = df.reset_index(drop=True)
            markers_df=df
        elif d['info']['name'][0] == 'Overwatch-VR':
            vr_data_df = pd.DataFrame(d['time_series'], columns=["vX","vY","vZ","vRoll","vJaw","vPitch"])
            vr_data_df['TimeStamp'] = d['time_stamps']

    return markers_df, vr_data_df
def process_markers(markers_df):
    markers_df = markers_df[markers_df['OverWatch_MainMarker'].apply(lambda x: x.isdigit() or x in ['START', 'STOP'])]
    markers_df['OverWatch_Trial'] = markers_df['OverWatch_MainMarker'].apply(lambda x: int(x) if x.isdigit() else x)
    return markers_df

def assign_trials_to_vr_data(vr_data_df, markers_df):
    # Asumiendo que 'Start' y 'Stop' están en los datos de 'Trial'
    start_df = markers_df[markers_df['OverWatch_MainMarker'] == 'START']
    stop_df = markers_df[markers_df['OverWatch_MainMarker'] == 'STOP']
    trial_intervals = []
    trial_numbers = []
    for start, stop in zip(start_df.itertuples(), stop_df.itertuples()):
        interval = pd.Interval(left=start.OverWatch_time_stamp, right=stop.OverWatch_time_stamp, closed='both')
        trial_intervals.append(interval)
        trial_numbers.append(
            start.OverWatch_Trial)  # Asumimos que el número del trial está correctamente en la fila de 'START'

    # Asociar cada punto de datos en vr_data_df con el intervalo correcto
    vr_data_df['True_OW_Trial'] = None  # Inicializar la columna para los números de trials
    for i, row in vr_data_df.iterrows():
        for interval, number in zip(trial_intervals, trial_numbers):
            if row['TimeStamp'] in interval:
                vr_data_df.at[i, 'True_OW_Trial'] = number
                break
    return vr_data_df

# Ejecutar el proceso CENTRAL
sorted_subjects_data = sorted(subjects_data, key=lambda x: x[0])
#sorted_subjects = sorted(subjects_data.keys())
for subject in subjects_data:
    for modality in subjects_data[subject]:
        for xdf_file in subjects_data[subject][modality]:
            print('Procesando: ',subject,modality,xdf_file)
            markers_df, vr_data_df = extraerData_xdf_file(xdf_file)
            processed_markers_df = process_markers(markers_df)
            vr_data_with_trials = assign_trials_to_vr_data(vr_data_df, markers_df)

            print('debug')
