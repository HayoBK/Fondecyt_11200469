#-----------------------------------------------------
#
#   Una nueva era Comienza, Agosto 2024
#   Por fin jugaremos con TODOS los DATOS
#   GAME is ON - Preparación final para Barany 2024... y tengo como 1 dias.
#   Version En AEROPUERTOS
#   PUPIL LABS INFO
#-----------------------------------------------------


#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import socket
from tqdm import tqdm
import time
from pathlib import Path

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
    home="D:/Titan-OneDrive"
    home_path = Path("D:/Titan-OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/Barany2024/"
    # Directorios version 2024 Agosto 22
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling_2/Outputs/Barany2024/"

if nombre_host == 'DESKTOP-PQ9KP6K':  #Remake por situaci´ón de emergencia de internet
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"
    Py_Processing_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Py_Processing/"
    Output_Dir = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/PyPro_traveling/Outputs/Barany2024/"

if nombre_host == 'Hayos-MacBook-Pro.local':
    home = str(Path.home())
    Py_Processing_Dir = home + "/Py_Adventure/PyPro_traveling_3/Py_Processing/"
    Output_Dir = home + "/Py_Adventure/PyPro_traveling_3/Outputs/Barany2024_Air/"
print('Compu identificado.')