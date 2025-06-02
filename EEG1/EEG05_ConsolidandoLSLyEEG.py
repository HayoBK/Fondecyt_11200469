#%%
#----------------------------------------------------------------------------------
# Script para consolidar todos mis datos de EEG para analisis.
#----------------------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import os
import sys
import mne
import pandas as pd

# Confirmación de ruta actual
print("Current working directory:", os.getcwd())

# Asegurar que EEG1 está en el path
eeg1_dir = os.path.abspath('./EEG1')
print("Intentando agregar:", eeg1_dir)
if eeg1_dir not in sys.path:
    sys.path.append(eeg1_dir)

import EEG00_HMod

# Rutas base
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# Listado de sujetos
lista_sujetos = [f for f in os.listdir(Py_Sujetos_Dir) if
                 f.startswith('P') and os.path.isdir(os.path.join(Py_Sujetos_Dir, f))]

# Log de errores o estados
log_script = []

# Diccionario para guardar cada raw asociado al sujeto
path_raws_por_sujeto = {}

# Lista de sujetos válidos
sujetos_validos = []


def corregir_vhdr(vhdr_path):
    base_dir = os.path.dirname(vhdr_path)
    vhdr_name = os.path.basename(vhdr_path)
    basename = os.path.splitext(vhdr_name)[0]

    eeg_filename = basename + '.eeg'
    vmrk_filename = basename + '.vmrk'

    with open(vhdr_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('DataFile='):
            new_lines.append(f'DataFile={eeg_filename}\n')
        elif line.startswith('MarkerFile='):
            new_lines.append(f'MarkerFile={vmrk_filename}\n')
        else:
            new_lines.append(line)

    with open(vhdr_path, 'w', encoding='utf-8') as file:
        file.writelines(new_lines)

    print(f"🔧 Corregido {vhdr_path} para apuntar a {eeg_filename} y {vmrk_filename}")

# Iterar por sujetos
for Sujeto in sorted(lista_sujetos):
    try:
        print(f"\n🔄 Procesando {Sujeto}...")
        Py_Specific_Dir = os.path.join(Py_Sujetos_Dir, Sujeto, 'EEG')

        # === Selección del archivo ===
        if Sujeto == 'P01':
            eegfile = os.path.join(Py_Specific_Dir, "P01_con_lagunas_cronologicas.fif")
        else:
            eegfile = os.path.join(Py_Specific_Dir, f"{Sujeto}_NAVI.vhdr")
            if os.path.exists(eegfile):
                corregir_vhdr(eegfile)

        # === Carga del archivo ===
        if not os.path.exists(eegfile):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eegfile}")

        if eegfile.endswith('.fif'):
            raw = mne.io.read_raw_fif(eegfile, preload=True)
        elif eegfile.endswith('.vhdr'):
            raw = mne.io.read_raw_brainvision(eegfile, preload=True)
        else:
            raise ValueError(f"Formato de archivo desconocido: {eegfile}")

        # === Guardar raw ===
        path_raws_por_sujeto[Sujeto] = eegfile
        sujetos_validos.append(Sujeto)
        print(f"✅ {Sujeto}: Archivo EEG cargado correctamente.")

    except Exception as e:
        error_msg = f"❌ {Sujeto}: ERROR - {str(e)}"
        print(error_msg)
        log_script.append((Sujeto, str(e)))

# === Reporte final ===
print(f"\n📊 Total de sujetos válidos: {len(sujetos_validos)}")
print("✅ Sujetos válidos:", sujetos_validos)

#%%
import json

# Guardar path_raws_por_sujeto como .json
json_path = os.path.join(Py_Processing_Dir, "EEG05_paths_raws_por_sujeto.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(path_raws_por_sujeto, f, indent=4)
print(f"📁 Diccionario de rutas guardado en:\n- {json_path}")

# Guardar sujetos_validos como .txt
txt_path = os.path.join(Py_Processing_Dir, "EEG05_sujetos_validos.txt")
with open(txt_path, 'w', encoding='utf-8') as f:
    for sujeto in sujetos_validos:
        f.write(sujeto + '\n')
print(f"📁 Lista de sujetos válidos guardada en:\n- {txt_path}")