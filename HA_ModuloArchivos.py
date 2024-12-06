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

import socket
import glob2
import pyxdf
from io import StringIO
import sys
import os
import pandas as pd

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

    ruta = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/" + mi_path
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
                    e = stderr_capture + ' en ' + mi_path
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