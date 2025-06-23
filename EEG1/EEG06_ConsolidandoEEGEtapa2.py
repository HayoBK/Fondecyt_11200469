#%%
#----------------------------------------------------------------------------------
# Script para consolidar todos mis datos de EEG para analisis.
# HF2 deberia haber corrido antes de generar este archivo! y luego EEG04
#----------------------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import os
import sys
import mne
import pandas as pd
import json

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

json_path = os.path.join(Py_Processing_Dir, "EEG05_paths_raws_por_sujeto.json")
with open(json_path, 'r', encoding='utf-8') as f:
    path_raws_por_sujeto = json.load(f)
print(f"✅ Diccionario cargado: {len(path_raws_por_sujeto)} rutas EEG.")

# === Cargar sujetos_validos desde TXT ===
txt_path = os.path.join(Py_Processing_Dir, "EEG05_sujetos_validos.txt")
with open(txt_path, 'r', encoding='utf-8') as f:
    sujetos_validos = [line.strip() for line in f.readlines()]
print(f"✅ Lista de sujetos válidos cargada: {len(sujetos_validos)} sujetos.")

# === TEST MODE: Limitar a un solo sujeto específico ===
# Comenta esta línea para procesar todos los sujetos
sujetos_validos = ['P33']


# === Loop general por sujetos válidos ===
for sujeto in sujetos_validos:
    print(f"\n🧠 Procesando sujeto: {sujeto}")

    try:
        eeg_path = path_raws_por_sujeto[sujeto]

        # === Cargar raw ===
        if eeg_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(eeg_path, preload=True)
        elif eeg_path.endswith('.vhdr'):
            raw = mne.io.read_raw_brainvision(eeg_path, preload=True)
        else:
            raise ValueError(f"Formato no soportado: {eeg_path}")

        print(f"✅ Archivo EEG cargado para {sujeto}")

        # === Extraer anotaciones a DataFrame ===
        anotaciones = raw.annotations
        if anotaciones is None or len(anotaciones) == 0:
            print(f"⚠️ No se encontraron anotaciones para {sujeto}")
            continue

        OriginalAnotaciones_df, NewAnotaciones_df = EEG00_HMod.traducir_anotaciones_originales_EEG(anotaciones)
        EEGMarkers = NewAnotaciones_df.rename(columns={
            'onset': 'OverWatch_time_stamp',
            'description': 'OverWatch_MarkerA'
        })[['OverWatch_time_stamp', 'OverWatch_MarkerA']]
        EEGMarkers['OverWatch_MarkerA'] = EEGMarkers['OverWatch_MarkerA'].astype(str)
        EEGMarkers_filtrada = EEG00_HMod.filtrar_marcadores(EEGMarkers, 'EEG')
        EEGTrials_limpios = EEG00_HMod.limpiar_trials_eeg(EEGMarkers)
        Eventos_J_df = NewAnotaciones_df[
            NewAnotaciones_df['description'].notna() &
            NewAnotaciones_df['description'].str.startswith('J_')
            ].copy()        #print(f"🧭 Se extrajeron {len(Eventos_J_df)} eventos tipo 'J_' para {sujeto}.")

        # === Definir carpeta del sujeto ===
        Py_Specific_Dir = os.path.join(Py_Sujetos_Dir, sujeto, 'EEG')

        # === Definir rutas a los archivos esperados ===
        ni_path = os.path.join(Py_Specific_Dir, "trials_forMATLAB_NI.csv")
        rv_path = os.path.join(Py_Specific_Dir, "trials_forMATLAB_RV.csv")

        # === Intentar cargar los archivos ===
        trials_ni, trials_rv = None, None

        if os.path.exists(ni_path):
            trials_ni = pd.read_csv(ni_path)
            trials_ni['Modalidad'] = 'NI'
            print(f"✅ {sujeto}: Archivo NI cargado con {len(trials_ni)} filas.")
        else:
            print(f"⚠️ {sujeto}: No se encontró el archivo NI: {ni_path}")

        if os.path.exists(rv_path):
            trials_rv = pd.read_csv(rv_path)
            trials_rv['Modalidad'] = 'RV'
            print(f"✅ {sujeto}: Archivo RV cargado con {len(trials_rv)} filas.")
        else:
            print(f"⚠️ {sujeto}: No se encontró el archivo RV: {rv_path}")

        # === Combinar si al menos uno existe ===
        if trials_ni is not None or trials_rv is not None:
            LSL_df_trials_combinado = pd.concat(
                [df for df in [trials_ni, trials_rv] if df is not None],
                ignore_index=True
            )
            print(f"🧩 {sujeto}: DataFrame combinado contiene {len(LSL_df_trials_combinado)} trials.")
        else:
            print(f"❌ {sujeto}: Ningún archivo de trials encontrado. Se omite el procesamiento dependiente.")
            LSL_df_trials_combinado = None  # para manejar lógica condicional más adelante

        # 2025.06.23
        # Tengo entonces (LSL_df_trials_combinado) & (EEGTrials_limpios)
        # (TASK 01) : Quiero compararlos, ver el nivel de sincronización entre ambos y evaluar si hay uno
        # mejor que el otro para definir con cual me quedo y cual ocupar en el archivo final de EEG raw markers
        # Luego (TASK 02)

    except Exception as e:
        print(f"❌ ERROR en {sujeto}: {str(e)}")