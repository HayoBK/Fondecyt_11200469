import os
import shutil
import socket
from pathlib import Path

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


# Directorio base donde se encuentran las carpetas P01, P02, ..., P53
base_dir = base_path

# Directorio destino en E:
dest_base_dir = 'E:/Pupils/'

# Iterar sobre las carpetas de P01 a P53
for i in range(1, 54):
    # Crear el nombre de la carpeta
    folder_name = f'P{i:02}'

    # Crear las rutas completas
    source_path = os.path.join(base_dir, folder_name, 'PUPIL_LAB')
    dest_path = os.path.join(dest_base_dir, folder_name)

    # Asegúrate de que el destino existe
    os.makedirs(dest_path, exist_ok=True)

    # Copiar el contenido de PUPIL_LAB al destino
    if os.path.exists(source_path):
        for item in os.listdir(source_path):
            s = os.path.join(source_path, item)
            d = os.path.join(dest_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    print(folder_name,' --> Listo')

print("Contenido copiado exitosamente. ")