import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro
import socket

home= str(Path.home())
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
Fenrir_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/006-Writing/04 - Paper Fondecyt 1/DataFrames/"
file = Fenrir_Processing_Dir+'CODEX_FENRIR.xlsx'

# ------------------------------------------------------------
#Identificar primero en que computador estamos trabajando
#-------------------------------------------------------------
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

file = Fenrir_Processing_Dir+'CODEX_FENRIR.xlsx'


df = pd.read_excel(file, index_col=0) #Aqui cargamos tu base de datos en la variable "df"

def keep_first_letter(text):
    return text[0] if isinstance(text, str) and len(text) > 0 else text
def keep_first_3letter(text):
    return text[0:3] if isinstance(text, str) and len(text) > 0 else text

# BDI
file =Fenrir_Processing_Dir+'BDI.csv'
#df_BDI = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_BDI = pd.read_csv(file,encoding='utf-8',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador

del df_BDI[df_BDI.columns[0]]
df_BDI = df_BDI.applymap(keep_first_letter)
df_BDI = df_BDI.apply(pd.to_numeric, errors='coerce')
df_BDI['BDI'] = df_BDI.sum(axis=1)
df_BDI=df_BDI[['BDI']]

df_merged = pd.merge(df,df_BDI,left_index=True,right_index=True, how='left')

# DHI
file =Fenrir_Processing_Dir+'DHI.csv'
#df_N = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_N = pd.read_csv(file,encoding='utf-8',index_col=1)
del df_N[df_N.columns[0]]
df_N.columns = [*df_N.columns[:-1], 'EVA']
df_N = df_N.applymap(keep_first_letter)
df_N = df_N.replace('S','4')
df_N = df_N.replace('A','2')
df_N = df_N.replace('N','0')
df_N = df_N.apply(pd.to_numeric, errors='coerce')
df_N['DHI'] = df_N.sum(axis=1)
df_N=df_N[['DHI','EVA']]

df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

# Edinburgo
file =Fenrir_Processing_Dir+'EDINBURGO.csv'
#df_N = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_N = pd.read_csv(file,encoding='utf-8',index_col=1)
del df_N[df_N.columns[0]]
df_N = df_N.applymap(keep_first_3letter)
value_to_score_DER = {'DER': 2, 'Der': 1, 'Pue': 1, 'Izq': 0, 'IZQ': 0}
value_to_score_IZQ = {'DER': 0, 'Der': 0, 'Pue': 1, 'Izq': 1, 'IZQ': 2}

def calculate_score(row, value_to_score):
    return sum(value_to_score[value] for value in row)

df_N['D'] = df_N.apply(lambda row: calculate_score(row, value_to_score_DER), axis=1)
popped_column = df_N.pop('D')
df_N['I'] = df_N.apply(lambda row: calculate_score(row, value_to_score_IZQ), axis=1)
df_N['D'] = popped_column
df_N['Edinburgo'] = ((df_N['D'] - df_N['I']) / (df_N['D'] + df_N['I'])) * 100

df_N=df_N[['Edinburgo']]

df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

#Niigata
file =Fenrir_Processing_Dir+'NIIGATA.csv'
#df_N = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_N = pd.read_csv(file,encoding='utf-8',index_col=1)
del df_N[df_N.columns[0]]

df_N['Niigata'] = df_N.sum(axis=1)
df_N=df_N[['Niigata']]
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

#STAI_A
file =Fenrir_Processing_Dir+'STAI_A.csv'
#df_N = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_N = pd.read_csv(file,encoding='utf-8',index_col=1)
del df_N[df_N.columns[0]]
df_N = df_N.applymap(keep_first_letter)
df_N = df_N.apply(pd.to_numeric, errors='coerce')
df_N['STAI_Rasgo'] = df_N.sum(axis=1)
df_N=df_N[['STAI_Rasgo']]
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

#STAI_E
file =Fenrir_Processing_Dir+'STAI_E.csv'
#df_N = pd.read_csv(file,sep=';',index_col=1) #sep es porque un csv procesado en NUMBERS de MAC usa ; en vez de , como separador
df_N = pd.read_csv(file,encoding='utf-8',index_col=1)
del df_N[df_N.columns[2]]
del df_N[df_N.columns[0]]
df_N = df_N.applymap(keep_first_letter)
df_N = df_N.apply(pd.to_numeric, errors='coerce')
df_N['STAI_Estado'] = df_N.sum(axis=1)
df_N=df_N[['STAI_Estado']]
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')


# Neurocognitivo
file =Fenrir_Processing_Dir+'NeuroCognitivo.xlsx'
df_N = pd.read_excel(file,index_col=0)
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

# Vestibular
file =Fenrir_Processing_Dir+'VEST.xlsx'
df_N = pd.read_excel(file,index_col=0)
df_N = df_N.applymap(keep_first_letter)
df_N = df_N.apply(pd.to_numeric, errors='coerce')
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

# SRQ
file =Fenrir_Processing_Dir+'SRQ_c.xlsx'
df_N = pd.read_excel(file,index_col=1)
del df_N[df_N.columns[0]]
df_merged = pd.merge(df_merged,df_N,left_index=True,right_index=True, how='left')

df_merged.to_excel((Fenrir_Processing_Dir + 'INFINITE_CODEX.xlsx'))
print('Done')
