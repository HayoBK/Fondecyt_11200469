# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022   --> Version Updated MSI 2024.11.29
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
# Procesar formulario SRQ para tesis Rosario Garrido
# ---------------------------------------------------------

#%%
import pandas as pd     #Base de datos
import numpy as np      # Libreria de calculos científicos
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro
import socket

#home= str(Path.home())
#Fenrir_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/006-Writing/04 - Paper Fondecyt 1/DataFrames/"
#file = Fenrir_Processing_Dir+'SRQ.csv'
#data = pd.read_csv(file,sep=';', index_col=0)

print('H-Identifiquemos compu... ')
nombre_host = socket.gethostname()
print(nombre_host)

if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"

if nombre_host == 'MSI':
    print('Estamos ok con ', nombre_host)
    home="D:/Titan-OneDrive"
    home_path = Path("D:/Titan-OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/Barany2024/"
    # Directorios version 2024 Agosto 22
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Outputs/Barany2024/"
    Fenrir_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_INFINITE/df_PsicoCognitivo/"

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"


file = Fenrir_Processing_Dir + 'SRQ.csv'
#data = pd.read_csv(file,sep=';', index_col=0)
data = pd.read_csv(file, index_col=0, encoding='utf-8')

    # Aqui leo el CSV que descargué de Google Forms. Ojo que le cambié el nombre para dejar marcada
    # la fecha en que lo bajé.
data.columns.values[0]='User'
data.columns.values[1]='PXX'
    # Aqui cambié el nombre de las primeras dos columna, para hacer más facil el manejarla

for i in range(58):
    data.columns.values[i+2]=str('Q_'+str(i+1))
    # Aqui cambié el nombre de cada una de las columnas de las preguntas por un codigo más fácil de manejar: Q_12 por ejemplo

User = data.pop('User')
    # Aqui "saqué por un rato la columna User porque no me importa

data = data.dropna()
    # Aqui saque todas las filas que tienen al menos una casilla vacia (sacando las que no se identificaron con PXX en la encuesta
#%%
#data = data.replace(regex=[r'\D+'], value="").astype(int)
    # Aqui elimine todo el texto que no sea numeros de la base de dato. Simplifica mucho todo

data = data.replace(regex=[r'\D+'], value="")

# Reemplazar valores vacíos o nulos con NaN
data = data.replace('', np.nan)

# Verifica si aún hay valores vacíos
print(data.isnull().sum())

# Convertir a tipo entero, ignorando las filas con NaN
data = data.astype('Int64')  # Usa el tipo Int64 que soporta NaN
#%%
data = data.dropna(subset=['PXX'])


def correct_PXX(p):
    corrected = 'P'+"{:02d}".format(p)
    return corrected

data['PXX'] = data['PXX'].apply(correct_PXX)
    # aqui lo que hice fue devolverle a la columan PXX el formato tipo "P05" al codigo del pacientes, para mantener
    # ese tipo de formato en todos lados

AversiveList = [2,5,6,7,11,14,15,18,19,20,22,23,24,26,28,29,30,31,32,33,36,39,40,41,44,45,46,48,50,52,54] # Lista de preguntas Q que corresponden a Aversivo.
HedonicList = [1,3,4,8,9,10,12,13,16,17,25,27,34,35,37,38,42,43,47,49,51,53,55,56,57,58]

AversiveResults = []
AversiveCategoria = []
HedonicResults = []
HedonicCategoria = []
#Estas listas estan vacias, es para ir agregando el resultado de nuestros datos.

for index, row in data.iterrows():
    #con ese loop vamos linea por linea de nuestra base de datos haciendo calculos. Muy util
    Aversive=0
    for A in AversiveList:
        # Con esto vamos a ir por cada elemento en la lista AversiveList
        Columna = "Q_"+ str(A) # Asi, por cada elemento en A, por ejemplo "2" terminamos con Q_2
        Aversive += row[Columna] #Suma a Aversive, el contenido de la columna A en la fila de un paciente en particular
    Aversive = Aversive/len(AversiveList) # aqui dividimos el puntaje sumado por la cantidad de elementos en la lista... ergo, el promedio
    AversiveResults.append(Aversive) # Añadimos a Esta lista de resultados en puntaje de Aversive
    if Aversive > 1.87:
        AversiveCategoria.append(True) # Aqui le pondremos True si supero el punto de corte del excel que me enviaste
    else:
        AversiveCategoria.append(False)

    #Ahora repitamos todo para Hedonico.

    Hedonic = 0
    for H in HedonicList:
        # Con esto vamos a ir por cada elemento en la lista AversiveList
        Columna = "Q_" + str(H)  # Asi, por cada elemento en A, por ejemplo "2" terminamos con Q_2
        Hedonic += row[Columna]  # Suma a Aversive, el contenido de la columna A en la fila de un paciente en particular
    Hedonic = Hedonic / len(HedonicList)  # aqui dividimos el puntaje sumado por la cantidad de elementos en la lista... ergo, el promedio
    HedonicResults.append(Hedonic)  # Añadimos a Esta lista de resultados en puntaje de Aversive
    if Hedonic > 2.1:
        HedonicCategoria.append(True)  # Aqui le pondremos True si supero el punto de corte del excel que me enviaste
    else:
        HedonicCategoria.append(False)

data['Col'] = HedonicCategoria #Añadimos la lista HedonicCategoria como columna a los datos
Col = data.pop('Col') #Sacamos la nueva columna y....
#data.insert(1,"Categoria Hedónico",Col) #La ponemos en el lugar 1, que es en realidad el segundo lugar... la primera columna es siempre la columna 0

# Y repetimos...
data['Col'] = HedonicResults
Col = data.pop('Col')
data.insert(1,"Puntaje Hedónico",Col)

data['Col'] = AversiveCategoria
Col = data.pop('Col')
#data.insert(1,"Categoria Aversivo", Col)

data['Col'] = AversiveResults
Col = data.pop('Col')
data.insert(1,"Puntaje Aversivo", Col)
data = data.iloc[:, :3]
data.to_excel(Fenrir_Processing_Dir+'SRQ_c.xlsx')

print(data)