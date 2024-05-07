# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------


# El desafío es poder procesar lo que emite Simian...
# AB_SimianMaze_A1_Grab_Motion_files.py recogerá los datos en una sola mega DataFRAME

import json
from pathlib import Path
import pandas as pd
import glob2
import os
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


def read_Navi_Motion_File(archivo):
    # Abrir el archivo ".motion"... aqui lo modifique ya a .json
    with open(archivo, "r") as read_file:
        data = json.load(read_file)
    # ------------------------------------------------------------------------------------------------------
    # Ahora vamos des jsoneando el archivo para dejarlo en una base amigable para lo que es Pandas.
    # Esto está hecho ESPECIFICAMENTE para los archivos que emite
    df = pd.json_normalize(data, 'motionTrackDataPerTrials')
    df['Trial'] = range(len(df))
    # Mini Codigo para cambiar orden de columnas en un DataFrame de Pandas.
    order = [4, 0, 1, 2, 3]  # setting column's order
    df = df[[df.columns[i] for i in order]]
    # explode all columns with lists of dicts
    df = df.apply(lambda x: x.explode()).reset_index(drop=True)
    df.rename(columns={'positions': 'P'}, inplace=True)
    cols_to_normalize = ['P']

    # if there are keys, which will become column names, overlap with excising column names
    # add the current column name as a prefix
    normalized = list()
    for col in cols_to_normalize:
        d = pd.json_normalize(df[col], sep='_')
        d.columns = [f'{col}_{v}' for v in d.columns]
        normalized.append(d.copy())

    # combine df with the normalized columns
    df = pd.concat([df] + normalized, axis=1).drop(columns=cols_to_normalize)
    PosColumn = df.groupby(['Trial']).cumcount()
    PosCluymn = pd.Series(PosColumn)
    df = df.assign(Positions=PosCluymn)
    order = [0, 7, 4, 5, 6, 1, 2, 3]  # setting column's order
    df = df[[df.columns[i] for i in order]]

    # ------------------------------------------------------------------------------------------------------

    # Ahora transformaremos los datos de posición y tamaño de plataforma a "Pool Diameter"
    # como unidad de medida. Después de un analisis de un ensayo "vuelta olímpica, determinamos que
    # el Diametro de la piscina son 280x2 = 560 unidades virturales asi que:
    def normalizar_a_PD(uvirtuales):
        return (uvirtuales / 560)

    df[['P_position_x', 'P_position_y', 'platformPosition.x', 'platformPosition.y']] = df[['P_position_x', 'P_position_y', 'platformPosition.x', 'platformPosition.y']].apply(
        normalizar_a_PD)
    return df

def Label_TrueTrialNames(row): #Con las proximas dos funciones asignamos los nombres "reales" de los blockes y los trials a los Bloque sy trials generados por el archivo motion
    Trial_Name = 'No asignado - error'
    if (row['Bloque']=='A') and (row['Trial']==0):
        Trial_Name = 'FreeNav'
    if (row['Bloque']=='A') and (row['Trial']==1):
        Trial_Name = 'Training'
    if (row['Bloque']=='A') and (row['Trial']>1) and (row['Trial']<6):
        Trial_Name = 'VisibleTarget_1'
    if (row['Bloque']=='A') and (row['Trial']>5):
        Trial_Name = 'HiddenTarget_1'

    if (row['Bloque']=='B'):
        Trial_Name = 'HiddenTarget_2'

    if (row['Bloque']=='C'):
        Trial_Name = 'HiddenTarget_3'

    if (row['Bloque']=='D'):
        Trial_Name = 'VisibleTarget_2'

    if (row['Sujeto']=='P01'):

        Trial_Name = 'Diego'
        if (row['Bloque'] == 'A') and (row['Trial'] == 0):
            Trial_Name = 'FreeNav'
        if (row['Bloque'] == 'A') and (row['Trial'] == 1):
            Trial_Name = 'Training'
        if (row['Bloque'] == 'A') and (row['Trial'] > 1) and (row['Trial'] < 4):
            Trial_Name = 'VisibleTarget_1'
        if (row['Bloque'] == 'A') and (row['Trial'] > 3):
            Trial_Name = 'HiddenTarget_1'

        if (row['Bloque'] == 'B'):
            Trial_Name = 'HiddenTarget_2'

        if (row['Bloque'] == 'C'):
            Trial_Name = 'HiddenTarget_3'

        if (row['Bloque'] == 'D'):
            Trial_Name = 'VisibleTarget_2'

    return Trial_Name

def Label_TrueTrialNumber(row):
    TrueTrialNumber = 0 #implica no fue asignado!
    if (row['True_Block']=='FreeNav'):
        TrueTrialNumber = row['Trial']+1
    if (row['True_Block']=='Training'):
        TrueTrialNumber = 1
    if (row['True_Block']=='VisibleTarget_1'):
        TrueTrialNumber = row['Trial']-1
    if (row['True_Block']=='HiddenTarget_1'):
        TrueTrialNumber = row['Trial']-5
    if (row['True_Block']=='HiddenTarget_2'):
        TrueTrialNumber = row['Trial']+1
    if (row['True_Block']=='HiddenTarget_3'):
        TrueTrialNumber = row['Trial']+1
    if (row['True_Block']=='VisibleTarget_2'):
        TrueTrialNumber = row['Trial']+1
    if (row['Sujeto'] == 'P01'):
        if (row['True_Block']=='HiddenTarget_1'):
            TrueTrialNumber = row['Trial']-3
    return TrueTrialNumber

def GrabMotion(searchfiles):
    global Trial_uID
    Navi_files = glob2.glob(searchfiles)
    Navi_files = sorted(Navi_files) #Esto ordena los archivos alfabeticamente
    df_list =[]   #preparamos una lista de los DataFrames que emergeran de cada archivo .motion
    t_df = pd.DataFrame
    for Navi_f in Navi_files:
        t_df = read_Navi_Motion_File(Navi_f)  #Aqui llamamos a nuesto super FUNCTION que definimos al principio del archivo y que des-json-iza un .motion a un Pandas DataFrame
        head, tail = os.path.split(Navi_f)  #Esto es para obtener solo el nombre del archivo y perder su directorio
        Bloque = tail[5] #esto es super especifico. el caracter [5] del nombre del archivo motion es A,B,C o D describiendo el bloque que usamos.
        Trial_uID +=100
        t_df.insert(1,'Origen',tail) #incorporamos el nombre del archivo de origen al DataFrame
        t_df.insert(1,'Trial_Unique_ID',Trial_uID + t_df['Trial'])
        t_df.insert(0,'Bloque',Bloque) #incorporamos el bloque de origen al DataFrame
        df_list.append(t_df) #añadimos el Dataframe a la lista que teniamo
        print('Anexado:  ',Px,'-',Bloque,'-',tail) #solo un reporte de como va la cosa.
    if len(df_list)!=0:
        t_df = pd.concat(df_list) #juntamos todos los dataframes de un unico paciente.
        t_df.insert(0,'Sujeto',Px) #le añadimos una columna descriptora
    return t_df

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/" # Esto evidentemente varia. puede que no varié de compu a compu de Hayo
#----------------------------------------------------

Px_list = ['P06','P12','P01','P02','P03','P04','P05','P07','P08','P10','P11','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P28',
           'P29','P27','P30','P31','P32','P33','P34','P35','P40','P43','P44','P45','P46','P47','P48','P49','P51',
           'P36','P37','P38','P39','P41','P42','P50','P52','P53'] # Lista de pacientes a incorporar en el análisis.
# Horror Missing = P36;P37;P39;P41;P42
# Pendiente = P38
#----------------------------------------------------
Trial_uID = 0
df_full_list =[] # Lista que albergara el DataFrame completo de cada paciente
for Px in Px_list:
    #Primero de los No inmersivo
    Searchfiles = BaseDir + Px + '/SimianMaze_No_Inmersivo/*.motion' #Vamos a buscar los archivos Motion en el directorio de cada paciente
    partial_df = GrabMotion(Searchfiles)
    partial_df.insert(1, 'Modalidad', 'No Inmersivo')
    df_full_list.append(partial_df) #lo añadimos a la megalista de los dataframes
    #Luego repetimos en realidad virtual
    Searchfiles = BaseDir + Px + '/SimianMaze_R_Virtual/*.motion' #Vamos a buscar los archivos Motion en el directorio de cada paciente
    partial_df = GrabMotion(Searchfiles)
    partial_df.insert(1, 'Modalidad', 'Realidad Virtual')
    df_full_list.append(partial_df) #lo añadimos a la megalista de los dataframes

df = pd.concat(df_full_list) #y juntamos todos los dataframes de todos los pacientes en un solo df
df['True_Block'] = df.apply(lambda row: Label_TrueTrialNames(row), axis=1)
column = df.pop('True_Block')
df.insert(2,'True_Block',column)
df['True_Trial'] = df.apply(lambda row: Label_TrueTrialNumber(row), axis=1)
column = df.pop('True_Trial')
df.insert(3,'True_Trial',column)

df.to_csv((Py_Processing_Dir + 'AB_SimianMaze_Z1_RawMotion.csv'))  #y lo exportamos
# df.to_excel('AB_SimianMaze_Z1_RawMotion.xlsx')
print('Todo listo')


#e_df=df.groupby(['Sujeto','Modalidad','Trial Unique-ID','True Block','True Trial']).agg({'P_timeMilliseconds': ['max'], 'P_position_x':['var'],'P_position_y':['var']})
#e_df = e_df.reset_index()
#print(e_df)
