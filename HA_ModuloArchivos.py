#---------------------------------------------
# Por fin un modulo para tener la cosa más ordenada
# Para no tener que cambiar en cada nuevo script para cada computador
# distinto la ruta
#---------------------------------------------

#from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro
import socket


def Nombrar_HomePath(mi_path):
    # Un ejemplo de la ruta luego de La ruta base "002-LUCIEN/Py_INFINITE/df_PsicoCognitivo/"
    # Busca en que compu estamos
    # y luego genera la ruta en One-Drive hasta la carpeta del Fondecyt

    print('H-Identifiquemos en que computador estamos... ')
    nombre_host = socket.gethostname()
    print(nombre_host)

    if nombre_host == 'DESKTOP-PQ9KP6K':
        home = "D:/Mumin_UCh_OneDrive"

    if nombre_host == 'MSI':
        home = "D:/Titan-OneDrive"

    ruta = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/" + mi_path
    return ruta
