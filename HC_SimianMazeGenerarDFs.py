# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Diciembre - 02, Lunes.
# Script para procesar Datos Simian Maze
# -----------------------------------------------------------------------
# versión espero que definitiva del procesamiento de datos
# Este primer Script ocupa ya un modulo de ruta para no repetirme y
# hacer codigo eficiente
# Este primer Script tiene por fin terminar con los DataFrame de Simian
# de Navegación, tanto los datos de posición, pero valores resumenes,
# Incluyendo pero con más cosas qeu solo CSE
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd     #Base de datos
import numpy as np
import matplotlib.pyplot as plt    #Graficos
import seaborn as sns   #Estetica de gráficos
import math
import os
import glob
from scipy.stats import entropy

pd.options.mode.chained_assignment = None  # default='warn'

#from pathlib import Path

#home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
#Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")


m_df = pd.read_csv((Py_Processing_Dir+'B_SimianMaze_Z1_RawMotion.csv'), index_col=0)
m_df.rename(columns = {'platformPosition.x':'platformPosition_x', 'platformPosition.y':'platformPosition_y'}, inplace = True)
m_df = m_df[m_df.True_Trial < 8] #Aqui eliminamos esos primeros sujetos en que hicimios demasiados trials por bloques

def DEL_POSZERO(data): # ELiminamos todos los primeros momentos muertos antes que el sujeto empiece a moverse...
    Ban=[]
    First=0
    Moving=False
    x1=0
    y1=0
    for row in data.itertuples():
        if row.Trial_Unique_ID != First:
            Moving=False
            First=row.Trial_Unique_ID
            x1=row.P_position_x
            y1=row.P_position_y
        Distance = math.sqrt((row.P_position_x - x1)**2 + (row.P_position_y - y1)**2)
        if Distance > 0.005:    #Para los mapas de calor 0.25 / 0.075 en modo original
            Moving=True
        if Moving==False:
            Ban.append('1')
        if Moving==True:
            Ban.append('0')
    data['to_Ban']=Ban
    data = data.loc[data['to_Ban']=='0']
    data = data.drop('to_Ban', axis=1)
    return data


m_df = DEL_POSZERO(m_df)

for col in m_df.columns:
    print(col)
FirstRow=True
MegaList = []

for row in m_df.itertuples():
    if FirstRow: #En la primera linea de un Unique Trial
        AssesedTrial = row.Trial_Unique_ID
        FirstRow = False
        S=row.Sujeto
        M=row.Modalidad
        TB = row.True_Block
        TT = row.True_Trial
        ID = row.Trial_Unique_ID
        time = row.P_timeMilliseconds
        x = row.P_position_x
        y = row.P_position_y
        path_length = 0
        plat = row.platformExists
        platX = row.platformPosition_x
        platY = row.platformPosition_y

        #Cálculo de CSE
        Time_Track = 100
        x_Track = x
        y_Track = y
        Measures_Track = 1
        CSE = 0

    if AssesedTrial != row.Trial_Unique_ID:  #Luego de que terminó el unique trial (hay que repetir esto al final del Loop.
        FirstRow = True
        rowList = [S,M,TB,TT,ID,time,path_length,CSE,plat] #x,y,plat,platX,playY]
        MegaList.append(rowList)

    # y aquí ponemos lo que ocurre en cada linea intermedia
    time=row.P_timeMilliseconds # voy updateando time para que quede "max" al terminar
    path_length+=((x - row.P_position_x)**2 + (y - row.P_position_y)**2)**0.5
    x = row.P_position_x
    y = row.P_position_y

    # Cálculo de CSE
    x_Track += x
    y_Track += y
    Measures_Track += 1

    if time > Time_Track:
        Time_Track += 100
        AvgX = x_Track / Measures_Track
        AvgY = y_Track / Measures_Track
        CSE += ((AvgX - platX)**2 + (AvgY - platY)**2)**0.5


#Aqui tenemos que repetir el "rowList" para dar cuenta del ultimo trial del loop
rowList = [S,M,TB,TT,ID,time,path_length,CSE,plat] #x,y,plat,platX,playY]
MegaList.append(rowList)

short_df = pd.DataFrame(MegaList, columns = ['Sujeto','Modalidad','True_Block','True_Trial','Trial_Unique_ID','Duration(ms)','Path_length','CSE','Platform_exists'])

#Hacer primer barrido para elminar los trials que hay que borrar
Banish_List=[]
for row in short_df.itertuples():
    if row.Path_length < 0.0005:
        Banish_List.append(row.Trial_Unique_ID)

#Aqui añadimos los Unique_IDd trials a borrar manualmente
Banish_List.extend([1201,2000,2100,1901])
Banish_List.extend([2700,3000,3100,3300,3400,3404,3005])
Banish_List.extend([3600,2901,2902,2903,4500,8500,7600,9300,11711,13102,13100,13600,14100,14300,14500,14600])
#a partir de P15 en adelante
Banish_List.extend([15500,15900,15604,15605,16600,22400,23606,24600,25400,25500,27604,27405,28500])
#a partir de P29
Banish_List.extend([30009,30010,30011,30400,31007,30608,31101,31302,30501,30802,32200,32506,34000,33900,34400,35100,35300,35703,36500,36600,37006,37910])

#a_partir de P40 a P49 - 10 de agosto
Banish_List.extend([38912,39406,39700,42604,43400,43801,44000,45700,48200,48003,48104,48600,48104])

#Revisión manual cualitativo por errores registrados durante toma de datos
#Absolutos
Banish_List.extend([2309,3507,3106,2502,4010,4104,4801,4405,5801,6604,9402,1604,13702,16110,25006,26003,31403,28007,28012,28606,28905,29106,29108,29110,29112,29200,29202,33307,33308,33309,33310,33312,43102,43103,45701])
#Dudosos
Banish_List.extend([4006,4100,5200,301,8902,9704,2304,7106,9600,12300,12302,13700,14302,15105,16611,18205])
Banish_List.extend([18302,19009,198009,19806,20306,21204,25508,22600,24303,25201,25204,26500,28204,3106,31607,31700])
Banish_List.extend([46900,44800,46705,44001,43700,42505,38600,40200,31800,32310,32401,34601,36009,36709,40910,41105,40210,47501,39900])
#Ultima revisión Abril 2024
Banish_List.extend([49706,50600,50900,51001,53900,54100,54800])

#Aqui vamos a limpiar los Trials conde NaviVissible tengan un error
for row in short_df.itertuples():
    if ((row.True_Block == 'VisibleTarget_1') or (row.True_Block == 'VisibleTarget_2')) and (row.CSE > 33):
        Banish_List.extend([row.Trial_Unique_ID])





# Aqui vamos a hacer una lista de los trials que eliminamos.
Banished_short_df = short_df[short_df['Trial_Unique_ID'].isin(Banish_List)]
Banished_short_df.to_excel(Py_Processing_Dir+'AB_SimianMaze_Z2_Banished_NaviData.xlsx')

Banished_long_df = m_df[m_df['Trial_Unique_ID'].isin(Banish_List)]
Banished_long_df.to_excel(Py_Processing_Dir+'AB_SimianMaze_Z2_Banished_NaviDataLong.xlsx')


short_df = short_df[~short_df['Trial_Unique_ID'].isin(Banish_List)]
m_df = m_df[~m_df['Trial_Unique_ID'].isin(Banish_List)]

# Aqui sacamos a sujetos especificos que se salieron de parametros o no cumplieron criterios de inclusion.
#Banish_List =['P13']
#short_df = short_df[~short_df['Sujeto'].isin(Banish_List)]
#m_df = m_df[~m_df['Sujeto'].isin(Banish_List)]

safe_df = short_df # aqui estamos guardando la df para revisar cuales fueron los elementos eliminados por DropNa
short_df = short_df.dropna()
dropped_df = safe_df[~safe_df.index.isin(short_df.index)] # aqui vemos los elementos dropeados
dropped_df.to_excel(Py_Processing_Dir+'AB_SimianMaze_Z2_Dropped_NaviData.xlsx')

safe_df = m_df
m_df = m_df.dropna()
dropped_df = safe_df[~safe_df.index.isin(m_df.index)] # aqui vemos los elementos dropeados
dropped_df.to_excel(Py_Processing_Dir+'AB_SimianMaze_Z2_Dropped_NaviDataLong.xlsx')


#Limpieza completa.


#Iniciamos revisión manual de Trials repetidos por errores, para elegir que UniqueTrials añadir a la lista de Banish Manual.

e_df = short_df.groupby(['Sujeto','Modalidad','True_Block','True_Trial'])['Trial_Unique_ID'].apply(list).reset_index()
print(e_df)

print('Trials en conflicto!')
Conflict = False
for row in e_df.itertuples():
    if len(row.Trial_Unique_ID) > 1:
        Conflict = True
        print(row.Sujeto, row.Modalidad, row.True_Block, row.True_Trial, row.Trial_Unique_ID)
        show_df = m_df.loc[ (m_df['Sujeto']==row.Sujeto) & (m_df['Modalidad']==row.Modalidad) & (m_df['True_Block']==row.True_Block) & (m_df['True_Trial']==row.True_Trial)]
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="Trial_Unique_ID", data=show_df, linewidth=3, alpha=0.8, sort=False)  # palette = sns.color_palette('Blues', as_cmap = True),
        print('check')

        show_df = m_df.loc[(m_df['Sujeto'] == row.Sujeto) & (m_df['Modalidad'] == row.Modalidad) & (m_df['True_Block'] == row.True_Block) & (m_df['True_Trial'] == row.True_Trial)]
        show_df = show_df.reset_index()
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="Trial_Unique_ID", data=show_df, linewidth=3, alpha=0.8, palette = sns.color_palette('Blues', as_cmap = True),sort=False)
        plt.show()
        show_df = short_df.loc[(short_df['Sujeto'] == row.Sujeto) & (short_df['Modalidad'] == row.Modalidad) & (short_df['True_Block'] == row.True_Block) & (short_df['True_Trial']
                                                                                                                                                             ==
                                                                                                                                               row.True_Trial)]
        ax = sns.barplot(x="Trial_Unique_ID",y="Path_length", data=show_df)
        plt.show()
        print('check')
print('¿Hubo conflicto? --> ',Conflict)



#-------------------------------------------------------------------------------------------------------------
#Ahora quiero corregir algunos errores puntuales interpolando datos especificos en Rows faltantes....
#-------------------------------------------------------------------------------------------------------------


def Interpolar_Row(Data, Suj, Bloq, TT):  # Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[
        (Data['Modalidad'] == 'No Inmersivo') &
        (Data['Sujeto'] == Suj) &
        (Data['True_Block'] == Bloq) &
        (Data['True_Trial'] == TT - 1)
        ]  # Obtenemos la línea inmediatamente anterior
    Ind = Ind[0]  # Esto lo hacemos para transformar una lista en un número
    print('Intentando realizar interpolación de ', Suj, Bloq, TT)
    print('Obteniendo índice (debe ser un solo número) --> ', Ind)

    # Crear una copia de la fila para modificarla
    Copy_Row = Data.iloc[[Ind]].copy()
    Promedio_CSE = (Data.iloc[Ind]['CSE'] + Data.iloc[Ind + 1]['CSE']) / 2
    Copy_Row.loc[:, 'CSE'] = Promedio_CSE
    Copy_Row.loc[:, 'True_Trial'] = TT
    print(Copy_Row)

    # Dividir el DataFrame en dos partes
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]

    # Concatenar las partes con la nueva fila interpolada
    Data = pd.concat([DataA, Copy_Row, DataB]).reset_index(drop=True)

    return Data


short_df = Interpolar_Row(short_df,'P12','HiddenTarget_3',2)
#short_df = Interpolar_Row(short_df,'P07','VisibleTarget_2',3)
#short_df = Interpolar_Row(short_df,'P19','HiddenTarget_3',3)
short_df = Interpolar_Row(short_df,'P21','HiddenTarget_3',2)
short_df = Interpolar_Row(short_df,'P28','HiddenTarget_3',2)

# Caso especial que no se puede usar Interpolar Row definido previamente:
#-------------------------------------------------------------------------------------------------------------
# Aqui se puede poner un primer TT

def Interpolar_Row_TT1_Missing(Data, Suj, Bloq, TT):  # Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[
        (Data['Modalidad'] == 'No Inmersivo') &
        (Data['Sujeto'] == Suj) &
        (Data['True_Block'] == Bloq) &
        (Data['True_Trial'] == TT + 1)
        ]  # Obtenemos la línea inmediatamente siguiente en este caso
    Ind = Ind[0]  # Transformamos una lista en un número
    print('Intentando realizar interpolación de ', Suj, Bloq, TT)
    print('Obteniendo índice (debe ser un solo número) --> ', Ind)

    # Crear una copia de la fila para modificarla
    Copy_Row = Data.iloc[[Ind]].copy()
    Promedio_CSE = Data.iloc[Ind]['CSE']
    Copy_Row.loc[:, 'CSE'] = Promedio_CSE
    Copy_Row.loc[:, 'True_Trial'] = TT
    print(Copy_Row)

    # Dividir el DataFrame en dos partes
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]

    # Concatenar las partes con la nueva fila interpolada
    Data = pd.concat([DataA, Copy_Row, DataB]).reset_index(drop=True)

    return Data


#short_df = Interpolar_Row_TT1_Missing(short_df,'P21','VisibleTarget_2',1)
short_df = Interpolar_Row_TT1_Missing(short_df,'P11','HiddenTarget_3',1)


def Interpolar_Row_TT7_Missing(Data, Suj, Bloq, TT):  # Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[
        (Data['Modalidad'] == 'No Inmersivo') &
        (Data['Sujeto'] == Suj) &
        (Data['True_Block'] == Bloq) &
        (Data['True_Trial'] == TT - 1)
        ]  # Obtenemos la línea inmediatamente anterior
    Ind = Ind[0]  # Transformamos una lista en un número
    print('Intentando realizar interpolación de ', Suj, Bloq, TT)
    print('Obteniendo índice (debe ser un solo número) --> ', Ind)

    # Crear una copia de la fila para modificarla
    Copy_Row = Data.iloc[[Ind]].copy()
    Promedio_CSE = Data.iloc[Ind]['CSE']
    Copy_Row.loc[:, 'CSE'] = Promedio_CSE
    Copy_Row.loc[:, 'True_Trial'] = TT
    print(Copy_Row)

    # Dividir el DataFrame en dos partes
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]

    # Concatenar las partes con la nueva fila interpolada
    Data = pd.concat([DataA, Copy_Row, DataB]).reset_index(drop=True)

    return Data


#short_df = Interpolar_Row_TT7_Missing(short_df,'P11','HiddenTarget_3',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P23','HiddenTarget_3',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P24','HiddenTarget_3',7)

short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_1',3)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_1',4)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',4)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',5)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',6)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',7)
#short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',5)
#short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',6)
#short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',7)
#short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_2',3)



#Repetir V2 desde V1 en P05
"""
Data= short_df
Data = Data.reset_index(drop=True)
Copy_Row = Data.loc[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='VisibleTarget_1') & (Data['True_Trial']<4)]
Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='VisibleTarget_1') & (Data['True_Trial']==1)]
Ind = Ind[0]
for i in range(3):
    Copy_Row.at[(Ind+i), 'True_Block'] = 'VisibleTarget_2'
Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='HiddenTarget_3') & (Data['True_Trial']==7)]
Ind = Ind[0]+1
DataA = short_df.iloc[:Ind, ]
DataB = short_df.iloc[Ind:, ]
short_df = DataA.append(Copy_Row).append(DataB).reset_index(drop=True)

"""
#-------------------------------------------------------------------------------------------------------------

# Ahora enriquecemos la base de datos con datos de los pacientes
# OJO ROSARIO ESTO LO PUEDES USAR PARA ENRIQUECER CON AVERSIVO/HEDONICO cualquier otro dato...
#-------------------------------------------------------------------------------------------------------------

codex = pd.read_excel((Py_Processing_Dir+'A_INFINITE_BASAL_DF.xlsx'),index_col=0)
Codex_Dict = codex.to_dict('series')
short_df['Edad'] = short_df['Sujeto']
short_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
m_df['Edad'] = m_df['Sujeto']
m_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
short_df['Genero'] = short_df['Sujeto']
short_df['Genero'].replace(Codex_Dict['Genero'], inplace=True)
m_df['Genero'] = m_df['Sujeto']
m_df['Genero'].replace(Codex_Dict['Genero'], inplace=True)
short_df['Dg'] = short_df['Sujeto']
short_df['Dg'].replace(Codex_Dict['Dg'], inplace=True)
m_df['Dg'] = m_df['Sujeto']
m_df['Dg'].replace(Codex_Dict['Dg'], inplace=True)
short_df['Grupo'] = short_df['Sujeto']
short_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)
move = short_df.pop('Grupo')
short_df.insert(1,'Grupo',move)
m_df['Grupo'] = m_df['Sujeto']
m_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)
move = m_df.pop('Grupo')
m_df.insert(1,'Grupo',move)
print(short_df)

#Aqui hare una DataBase Limpia de los sujetos incluidos en el estudio, solo datos básicos
PXX_List = short_df['Sujeto'].unique()  # Obtenemos una lista de todos los sujetos individualizados
codex_df = pd.DataFrame(PXX_List, columns = ['Sujeto'])
codex_df['Edad'] = codex_df['Sujeto']
codex_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
codex_df['Grupo'] = codex_df['Sujeto']
codex_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)

#Resumen_codex_df = codex_df.groupby('Grupo')['Edad'].agg(Conteo='size', Edad_promedio='mean').reset_index()

#codex_df.to_csv(Py_Processing_Dir+'AB_SimianMaze_Z4_Resumen_Pacientes_Analizados.csv')
#m_df.to_csv(Py_Processing_Dir+'C_SimianMaze_Z2_NaviData_con_posicion.csv')

#m_df.to_excel('AB_SimianMaze_Z2_NaviData_con_posicion.xlsx')

#short_df.to_csv(Py_Processing_Dir+'D_SimianMaze_Z3_NaviDataBreve_con_calculos.csv')

#short_df.to_excel('AB_SimianMaze_Z3_NaviDataBreve_con_calculos.xlsx')

output_dir = Py_Processing_Dir  # Asegúrate de que Py_Processing_Dir esté definido

# Buscar y eliminar archivos que comiencen con 'AB'
for file_path in glob.glob(os.path.join(output_dir, "AB*")):
    try:
        os.remove(file_path)
        print(f"Archivo eliminado: {file_path}")
    except Exception as e:
        print(f"Error al intentar eliminar {file_path}: {e}")

print('100% todo listo ')
durations = m_df.groupby('Trial_Unique_ID')['P_timeMilliseconds'].agg(Latencia=lambda x: x.max() - x.min())

# 2. Combinar las duraciones con el DataFrame resumen short_df
short_df = short_df.merge(durations, on='Trial_Unique_ID', how='left')

# 3. Exportar short_df a un archivo Excel
#output_file = Py_Processing_Dir+"FaundezDiciembre.xlsx"
#short_df.to_excel(output_file, index=False)

def calcular_distancia(x1, y1, x2, y2):
    """Calcula la distancia euclidiana entre dos puntos."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Métricas
def calcular_metricas_por_trial(m_df, short_df):
    resultados = []

    for trial_id in m_df['Trial_Unique_ID'].unique():
        trial_data = m_df[m_df['Trial_Unique_ID'] == trial_id].copy()
        trial_data = trial_data.sort_values('P_timeMilliseconds')  # Asegurar orden temporal

        # Variables necesarias
        platform_x, platform_y = trial_data.iloc[0][['platformPosition_x', 'platformPosition_y']]
        pos_x, pos_y = trial_data['P_position_x'].values, trial_data['P_position_y'].values
        timestamps = trial_data['P_timeMilliseconds'].values

        # 1. Velocidad promedio
        total_distance = np.sum(calcular_distancia(pos_x[:-1], pos_y[:-1], pos_x[1:], pos_y[1:]))
        total_time = (timestamps[-1] - timestamps[0]) / 1000  # Convertir a segundos
        velocidad_promedio = total_distance / total_time if total_time > 0 else 0

        # 2. Inmovilidad (freezing)
        velocidades = calcular_distancia(pos_x[:-1], pos_y[:-1], pos_x[1:], pos_y[1:]) / np.diff(timestamps)
        freezing = np.sum(velocidades < 0.001) / len(velocidades) if len(velocidades) > 0 else 0

        # 3. Índice de eficiencia
        distancia_directa = calcular_distancia(pos_x[0], pos_y[0], platform_x, platform_y)
        indice_eficiencia = distancia_directa / total_distance if total_distance > 0 else np.nan

        # 4. Entropía espacial
        hist, _, _ = np.histogram2d(pos_x, pos_y, bins=10, density=True)
        entropia_espacial = entropy(hist.ravel())

        # 5. Clasificación de estrategias (simplificada)
        if indice_eficiencia > 0.9:
            estrategia = 'Directa'
        elif freezing > 0.5:
            estrategia = 'Estática'
        elif total_distance > distancia_directa * 3:
            estrategia = 'Exploratoria'
        else:
            estrategia = 'Intermedia'

        # Agregar resultados
        resultados.append({
            'Trial_Unique_ID': trial_id,
            'Velocidad_Promedio': velocidad_promedio,
            'Freezing': freezing,
            'Indice_Eficiencia': indice_eficiencia,
            'Entropia_Espacial': entropia_espacial,
            'Estrategia_Simple': estrategia
        })

    # Convertir resultados a DataFrame
    resultados_df = pd.DataFrame(resultados)

    # Merge con short_df
    short_df = short_df.merge(resultados_df, on='Trial_Unique_ID', how='left')

    return short_df

short_df = calcular_metricas_por_trial(m_df, short_df)


def clasificar_estrategia(trial_data):
    """
    Clasifica la estrategia de búsqueda basada en parámetros del ensayo.

    Estrategias posibles:
    - Directa
    - Exploratoria
    - Circular
    - Perimetral (thigmotaxis)
    - Estática
    """
    # Coordenadas de inicio y plataforma
    platform_x, platform_y = trial_data.iloc[0][['platformPosition_x', 'platformPosition_y']]
    pos_x, pos_y = trial_data['P_position_x'].values, trial_data['P_position_y'].values

    # Distancia total recorrida
    distancia_total = np.sum(calcular_distancia(pos_x[:-1], pos_y[:-1], pos_x[1:], pos_y[1:]))

    # Distancia directa (inicio → plataforma)
    distancia_directa = calcular_distancia(pos_x[0], pos_y[0], platform_x, platform_y)

    # Índice de eficiencia
    indice_eficiencia = distancia_directa / distancia_total if distancia_total > 0 else np.nan

    # Tiempo en áreas perimetrales y cercanas
    radio_lab = max(pos_x.max() - pos_x.min(), pos_y.max() - pos_y.min()) / 2
    distancia_a_plataforma = calcular_distancia(pos_x, pos_y, platform_x, platform_y)
    tiempo_cerca_plataforma = np.sum(distancia_a_plataforma < radio_lab * 0.3) / len(distancia_a_plataforma)
    tiempo_en_perimetro = np.sum(distancia_a_plataforma > radio_lab * 0.7) / len(distancia_a_plataforma)

    # Clasificación basada en heurísticas
    if indice_eficiencia > 0.8:
        estrategia = "Directa"
    elif tiempo_cerca_plataforma > 0.5:
        estrategia = "Exploratoria"
    elif tiempo_en_perimetro > 0.5:
        estrategia = "Perimetral (thigmotaxis)"
    elif distancia_total > distancia_directa * 3:
        estrategia = "Circular"
    else:
        estrategia = "Estática"

    return estrategia


def agregar_estrategias(m_df, short_df):
    """
    Calcula estrategias de búsqueda para cada Trial_Unique_ID
    y las agrega a short_df.
    """
    estrategias = []
    for trial_id in m_df['Trial_Unique_ID'].unique():
        trial_data = m_df[m_df['Trial_Unique_ID'] == trial_id].copy()
        estrategia = clasificar_estrategia(trial_data)
        estrategias.append({'Trial_Unique_ID': trial_id, 'Estrategia': estrategia})

    estrategias_df = pd.DataFrame(estrategias)
    short_df = short_df.merge(estrategias_df, on='Trial_Unique_ID', how='left')

    return short_df

def calcular_entropias(trial_data):
    """
    Calcula Herror, Hpath y Htotal para un conjunto de datos de navegación.
    """
    # Coordenadas del sujeto y la plataforma
    pos_x, pos_y = trial_data['P_position_x'], trial_data['P_position_y']
    platform_x, platform_y = trial_data.iloc[0]['platformPosition_x'], trial_data.iloc[0]['platformPosition_y']

    # Distancias al objetivo (plataforma)
    distancias = np.sqrt((pos_x - platform_x) ** 2 + (pos_y - platform_y) ** 2)
    sigma_d = np.std(distancias)  # Desviación estándar de las distancias
    Herror = np.log(sigma_d) if sigma_d > 0 else 0

    # Calcular centroide del camino
    centroide_x, centroide_y = np.mean(pos_x), np.mean(pos_y)

    # Desviaciones estándar respecto al centroide (ejes de la elipse)
    sigma_a = np.std(pos_x)
    sigma_b = np.std(pos_y)
    Hpath = np.log(sigma_a * sigma_b) if sigma_a > 0 and sigma_b > 0 else 0

    # Entropía total
    Htotal = Herror + Hpath

    return Herror, Hpath, Htotal

def calcular_entropias_por_trial(m_df, short_df):
    """
    Calcula las entropías para todos los trials en m_df y las agrega a short_df.
    """
    # Lista para guardar los resultados
    resultados = []

    # Iterar sobre cada Trial_Unique_ID
    for trial_id in m_df['Trial_Unique_ID'].unique():
        trial_data = m_df[m_df['Trial_Unique_ID'] == trial_id]
        Herror, Hpath, Htotal = calcular_entropias(trial_data)
        resultados.append({
            'Trial_Unique_ID': trial_id,
            'Herror': Herror,
            'Hpath': Hpath,
            'Htotal': Htotal
        })

    # Convertir los resultados a DataFrame
    entropias_df = pd.DataFrame(resultados)

    # Combinar con short_df
    short_df = short_df.merge(entropias_df, on='Trial_Unique_ID', how='left')

    return short_df

# Ejemplo de uso
# Supongamos que m_df y short_df ya están definidos
# Calculamos las entropías y actualizamos short_df
short_df = calcular_entropias_por_trial(m_df, short_df)
# Ejemplo de uso
short_df = agregar_estrategias(m_df, short_df)
m_df.to_csv(Py_Processing_Dir+'C1_SimianMaze_Z2_PosXY.csv')
short_df.to_csv(Py_Processing_Dir+'C2_SimianMaze_Z3_Resumen_Short_df.csv')
print('Work is Done')

#%%
print('Terminamos')