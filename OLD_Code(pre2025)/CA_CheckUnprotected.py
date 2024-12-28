import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro

import socket
# Obtiene el nombre del host de la máquina actual

home= str(Path.home())
home=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"

base_path = Path.home() / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"

nombre_host = socket.gethostname()
print(nombre_host)
if nombre_host == 'DESKTOP-PQ9KP6K':
    home="D:/Mumin_UCh_OneDrive"
    home_path = Path("D:/Mumin_UCh_OneDrive")
    base_path= home_path / "OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS"

data = []

# Recorrer cada carpeta de sujeto (P01, P02, etc.)
for subject_folder in sorted(base_path.iterdir()):
    if subject_folder.is_dir():  # Verificar que sea una carpeta
        # Crear un diccionario para almacenar los datos de las subcarpetas
        subject_data = {'Subject': subject_folder.name}
        # Recorrer cada subcarpeta dentro de la carpeta del sujeto
        for subfolder in subject_folder.iterdir():
            if subfolder.is_dir():  # Verificar que sea una carpeta
                # Verificar si la subcarpeta está vacía
                is_empty = -1 if not any(subfolder.iterdir()) else 1
                # Agregar la información al diccionario
                subject_data[subfolder.name] = is_empty
        # Agregar el diccionario a la lista de datos
        data.append(subject_data)

# Crear un dataframe de Pandas con la información
df = pd.DataFrame(data)

# Rellenar los valores faltantes con -1, en caso de que haya sujetos sin algunas subcarpetas
df.fillna(-1, inplace=True)

# Exportar el dataframe a un archivo CSV
output_file = Path.home() / "subject_folders_info.xlsx"  # Cambia la ruta si es necesario
df.to_excel(output_file, index=False)
print(f"El archivo se ha guardado en: {output_file}")