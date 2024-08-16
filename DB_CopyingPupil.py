#%%
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

destino = home + "/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/004 - Alimento para LUCIEN/Pupil_Labs_RV/"

#%%
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

#%%
base_dir = base_path

# Directorio destino donde se reunirán los archivos CSV
dest_dir = destino

# Asegúrate de que el directorio destino exista
os.makedirs(dest_dir, exist_ok=True)

# Iterar sobre las carpetas de P01 a P53
for i in range(1, 54):
    # Crear el nombre del archivo a buscar
    file_name = f'P{i:02}_gaze_positions.csv'

    # Recorrer la estructura de carpetas buscando el archivo
    for root, dirs, files in os.walk(os.path.join(base_dir, f'P{i:02}')):
        if file_name in files:
            # Si se encuentra el archivo, se copia al destino
            source_file = os.path.join(root, file_name)
            dest_file = os.path.join(dest_dir, file_name)
            shutil.copy2(source_file, dest_file)
            print(f'Archivo {file_name} copiado a {dest_dir}')
            break  # Deja de buscar en esta carpeta, pues ya encontraste el archivo

print("Proceso completado.")


print("Contenido copiado exitosamente. ")