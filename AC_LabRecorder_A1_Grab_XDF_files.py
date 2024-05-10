# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------


# El desafío es poder procesar lo que emite Lab Recorder
#%%

import json
from pathlib import Path
import pandas as pd
import glob2
import os
import pyxdf
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
import numpy as np

from pathlib import Path

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"


home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"


Navi_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
Navi_df = Navi_df.loc[Navi_df['Modalidad']=='No Inmersivo']
print ('si?')

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

Navi_df = MWM_to_OW_trials(Navi_df)

print('hello hello')
# Aqui incluimos un csv de pupil Labs de prueba...
#test_df = pd.read_csv('test_gaze.csv')

def Extract(lst,place):
    return [item[place] for item in lst]

def ClearMarkers(df):

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

#Vamos a buscar todos los archivos de datos del Pupil Labs de Felipe
searchfiles = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/004 - Alimento para LUCIEN/Pupil_labs_Faundez/*.csv"
Pupil_files = glob2.glob(searchfiles) #obtiene una LISTA de todos los archivos que calcen con "searchfiles"
Pupil_files = sorted(Pupil_files) # los ordena alfabeticamente
PxxList =[]
Df_List = []
for Pupil_f in Pupil_files:
    head, tail = os.path.split(Pupil_f) #Esto es para obtener solo el nombre del archivo y perder su directorio
    CodigoPxx = str(tail[0:3])
    print('Adquiriendo datos de Pupil.csv para sujeto ',CodigoPxx)
    PxxList.append(CodigoPxx)
    t_df = pd.read_csv(Pupil_f)
    t_df.insert(0, 'Sujeto', CodigoPxx)

# Pensando en XDF-Labd Recorder Con esta función extraemos un elemento (en orden-place) de todo el Stream, un parametro... que están
# guardados en "time series" mientras que en "time stamps" esta la clave de sincronización de Lab Recorder
#Ahora vamos a por los archivos XDF de donde extraeremos los Timestamps a usar para analizar los archivos csv de PupilLabs

    Dir = BaseDir + CodigoPxx
    pattern = Dir + "/LSL_LAB/**/*NI*.xdf"
    XDF_files = glob2.glob(pattern)
    print('Deberia haber un file aqui: ')
    print(XDF_files)
    print('Ojalá...')

    for x in XDF_files:
        head, tail = os.path.split(x) #Esto es para obtener solo el nombre del archivo y perder su directorio
        print(CodigoPxx,tail)

# Aqui obtenemos un XDF de Lab Recorder de prueba
#FileName = BaseDir + 'P05/LSL_LAB/ses-NI/eeg/sub-P05_ses-NI_task-Default_run-001_eeg.xdf'

        # Extraemos los datos del XDF
        data, header = pyxdf.load_xdf(x)
        # data[0] es el Stream de Pupil Capture
        # data[1] es el Stream de Overwatch

        #t= data[0]['time_stamps']
        #x= Extract(data[0]['time_series'],1) # Stream pupilCapture, Canal 1: norm_pos_x
        #y= Extract(data[0]['time_series'],2)
        #LSL_df = pd.DataFrame(list(zip(t, x, y)), columns =['LSL_timestamp', 'LSL_norm_pos_x','LSL_norm_pos_x'])
        #LSL_df = LSL_df.loc[(LSL_df['LSL_timestamp']>3000) & (LSL_df['LSL_timestamp']<3100)]
        #ax = sns.lineplot(data= LSL_df, x= 'LSL_timestamp', y='LSL_norm_pos_x', alpha=0.3)
        #plt.show()

        for d in data:
            if d['info']['name'][0]=='Overwatch-Markers':

                time_stamp = d['time_stamps']
                MarkersAlfa = Extract(d['time_series'],0) # Stream OverWatch Markers, Canal 0: Marker Primario
                MarkersBeta = Extract(d['time_series'],1) # Stream pupilCapture, Canal 1: Marker Secundario
                Markers_df = pd.DataFrame(list(zip(time_stamp, MarkersAlfa, MarkersBeta)), columns =['OverWatch_time_stamp', 'OverWatch_MarkerA', 'OverWatch_MarkerB'])
                MarkersA_df = Markers_df.loc[Markers_df['OverWatch_MarkerA']!='NONE']
# -------------------------------------------------------------
# Vamos a iniciar el análisis de los Markers de OverWatch para
# quedar con una lista confiable de marcadores

        OverWatch_ClearedMarkers_df = ClearMarkers(MarkersA_df) # Aqui construimos la base de datos con los marcadores con todas las
            # ... correcciones interpretativas identificadas hasta el momento.
        df= OverWatch_ClearedMarkers_df
        df = df.reset_index(drop=True)
        df['ori']=tail
        if CodigoPxx == 'P13':
            df = df.loc[(df['OverWatch_time_stamp']<8000)]
        print('Si todo sale bien este numero debiese ser 66 -->', len(df.index))

        inicios = df[df['OverWatch_MainMarker'] == 'START']['OverWatch_time_stamp'].reset_index(drop=True)
        OW_Trials = df[df['OverWatch_MainMarker'] == 'START']['OverWatch_Trial'].reset_index(drop=True)
        finales = df[df['OverWatch_MainMarker'] == 'STOP']['OverWatch_time_stamp'].reset_index(drop=True)
        trials = pd.DataFrame({'Start':inicios, 'End':finales,'OW_trials':OW_Trials})

        #LSL_df.rename(columns = {'LSL_timestamp':'timestamp'},inplace=True)
        trial_labels = list(range(1,34))
        trial_labels = trials['OW_trials'].tolist() #Ojo aqui que quede bien...
        trial_labels = np.array(trial_labels)
        bins = pd.IntervalIndex.from_tuples(list(zip(trials['Start'], trials['End'])), closed = 'left')
        try_df = t_df
        try_df['OW_Trial'] = pd.cut(t_df['gaze_timestamp'],bins).map(dict(zip(bins,trial_labels)))
#LSL_df['OW_Trial_info'] = LSL_df['OW_Trial'].apply(lambda x: trials.iloc[x])
        codex = pd.read_excel((Py_Processing_Dir+'AB_OverWatch_Codex.xlsx'),index_col=0) # Aqui estoy cargando como DataFrame la lista de códigos que voy a usar, osea, los datos del diccionario. Es super
# imporatante el index_col=0 porque determina que la primera columna es el indice del diccionario, el valor que usaremos para guiar los reemplazos.
        Codex_Dict = codex.to_dict('series') # Aqui transformo esa Dataframe en una serie, que podré usar como diccionario
        try_df['MWM_Block'] = try_df['OW_Trial'] # Aqui empieza la magia, creo una nueva columna llamada MWM_Bloque, que es una simple copia de la columna OverWatch_Trial.
        # De
        # momento
        # son identicas
        try_df['MWM_Block'].replace(Codex_Dict['MWM_Bloque'], inplace=True) # Y aqui ocurre la magia: estoy reemplazando cada valor de la columna recien creada,
        # ocupando el diccionario que
# armamos como guia para hacer el reemplazo
        try_df['MWM_Trial'] = try_df['OW_Trial']
        try_df['MWM_Trial'].replace(Codex_Dict['MWM_Trial'], inplace=True)
        try_df['Ori_XDF']=tail
        try_df = try_df.dropna(subset=['OW_Trial'])
        try_df= try_df.reset_index(drop=True)
        try_df['first'] = (try_df.groupby('OW_Trial').cumcount() == 0).astype(int)
        Df_List.append(try_df)

        codex2 = pd.read_excel((Py_Processing_Dir+'AA_CODEX.xlsx'), index_col=0)
        Codex_Dict2 = codex2.to_dict('series')
        try_df['Edad'] = try_df['Sujeto']
        try_df['Edad'].replace(Codex_Dict2['Edad'], inplace=True)

        try_df['Grupo'] = try_df['Sujeto']
        try_df['Grupo'].replace(Codex_Dict2['Grupo'], inplace=True)
        move = try_df.pop('Grupo')
        try_df.insert(1, 'Grupo', move)
        try_df.rename(columns={'gaze_timestamp' : 'timestamp'}, inplace=True)
        a = try_df.pop('world_timestamp')

        ForCodex3 = Navi_df.loc[Navi_df['Sujeto']==CodigoPxx]
        ForCodex3 = ForCodex3.set_index('OW_trial')
        Codex3 = ForCodex3.to_dict('series')
        try_df['CSE'] = try_df['OW_Trial']
        try_df['CSE'].replace(Codex3['CSE'],inplace=True)

    print('Almost there...', CodigoPxx)
    print('Terminé con ',CodigoPxx)

final_df = pd.concat(Df_List)
final_df['Main_Block']=final_df['MWM_Block']  # Aqui vamos a recodificar los Bloques en un "Main Block" más grueso
final_df.loc[(final_df.Main_Block == 'FreeNav'),'Main_Block']='Non_relevant'
final_df.loc[(final_df.Main_Block == 'Training'),'Main_Block']='Non_relevant'
final_df.loc[(final_df.Main_Block == 'Rest_1'),'Main_Block']='Non_relevant'
final_df.loc[(final_df.Main_Block == 'Rest_2'),'Main_Block']='Non_relevant'
final_df.loc[(final_df.Main_Block == 'Rest_3'),'Main_Block']='Non_relevant'

final_df.loc[(final_df.Main_Block == 'VisibleTarget_1'),'Main_Block']='Target_is_Visible'
final_df.loc[(final_df.Main_Block == 'VisibleTarget_2'),'Main_Block']='Target_is_Visible'
final_df.loc[(final_df.Main_Block == 'HiddenTarget_1'),'Main_Block']='Target_is_Hidden'
final_df.loc[(final_df.Main_Block == 'HiddenTarget_2'),'Main_Block']='Target_is_Hidden'
final_df.loc[(final_df.Main_Block == 'HiddenTarget_3'),'Main_Block']='Target_is_Hidden'
final_df = final_df.loc[final_df['Sujeto']!='P13']
final_df = final_df.loc[final_df['Sujeto']!='P06']

final_df.to_csv(Py_Processing_Dir+'AC_PupilLabs_SyncData_Faundez.csv')
#%%



print('hola')