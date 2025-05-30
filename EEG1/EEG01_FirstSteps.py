# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2025 - Mayo - 12, Lunes.
# Veamos como nos va con EEG MNE.
# -----------------------------------------------------------------------

# Aprendizajes
# 1) MNE tiene una forma de cargar nativamente  BrainVision files de los que usa Billeke
# 2) MNE necesita además de cargar MNE, cargar la biblioteca PyQt5

#%%
import HA_ModuloArchivos as H_Mod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # <- Esto fuerza el backend Qt5 para visualización dinámica
import mne
import sys
import os

print("Current working directory:", os.getcwd())
eeg1_dir = os.path.abspath('./EEG1')  # O pon ruta absoluta manual

print("Intentando agregar:", eeg1_dir)
if eeg1_dir not in sys.path:
    sys.path.append(eeg1_dir)

import EEG00_HMod

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")
print('Chequeando si MNE está ok, (pregunto por su version): ', mne.__version__)

Sujeto = 'P33'

EEG00_HMod.generar_Sync_Markers_Files(Sujeto,H_Mod)  # Extramos de LSL los marcadores de Eventos
Py_Specific_Dir = Py_Sujetos_Dir + Sujeto + '/EEG/'
eegfile = Py_Specific_Dir + Sujeto + '_NAVI.vhdr'  # Cargamos el EEG File crudo

# Cargar los datos
raw = mne.io.read_raw_brainvision(eegfile, preload=True)

#Intentemos aqui trbajar con las anotaciones
annotations = raw.annotations
ori_ann_df, new_ann_df = EEG00_HMod.traducir_anotaciones_originales_EEG(annotations)
Markers_df = new_ann_df.rename(columns={
    'onset': 'OverWatch_time_stamp',
    'description': 'OverWatch_MarkerA'
})[['OverWatch_time_stamp', 'OverWatch_MarkerA']]
Markers_df['OverWatch_MarkerA'] = Markers_df['OverWatch_MarkerA'].astype(str)

exito, output = H_Mod.ClearMarkers_LEGACY(Markers_df)
output2 = H_Mod.LimpiarErroresdeOverwatch1(output)


resultados_sync = EEG00_HMod.calcular_delta_sync('P33', raw, H_Mod) # Calcular la diferencia de sincronización entre LSL y marcadores del EEG
print(resultados_sync)

delta_promedio_NI = resultados_sync['NI']['promedio']
delta_promedio_RV = resultados_sync['RV']['promedio']


print(delta_promedio_NI, delta_promedio_RV)
file= Py_Specific_Dir + 'trials_forMATLAB_NI.csv'
trials_by_LSL = pd.read_csv(file)
trials_by_LSL['start_time']+=delta_promedio_NI
trials_by_LSL['end_time']+=delta_promedio_NI
#%%
# Integrar los eventos de modalidad NI
raw, unique_trials_NI = EEG00_HMod.integrar_time_markers_en_raw(
    raw, Py_Specific_Dir,
    'trials_forMATLAB_NI.csv',
    'fixation_forMATLAB_NI.csv',
    'blinks_forMATLAB_NI.csv',
    delta_promedio=delta_promedio_NI,
    modalidad='NI',
    modo_operacion='solo'
)

# Integrar eventos de modalidad RV (si corresponde en el mismo EEG)
raw, unique_trials_RV = EEG00_HMod.integrar_time_markers_en_raw(
    raw, Py_Specific_Dir,
    'trials_forMATLAB_RV.csv',
    'fixation_forMATLAB_RV.csv',
    'blinks_forMATLAB_RV.csv',
    delta_promedio=delta_promedio_RV,
    modalidad='RV',
    modo_operacion='add'
)
df_annot = raw.annotations.to_data_frame()
print(df_annot.head())
# Guardar el EEG anotado


raw.save(Py_Specific_Dir + f"{Sujeto}_conLSLMarkers_eeg.fif", overwrite=True)

events, event_id = mne.events_from_annotations(raw)
print(events[:5])
print(event_id)

# Verifica info general
print(raw.info)
print("Backend matplotlib:", matplotlib.get_backend())

# Visualiza los datos crudos
raw.plot(n_channels=64, block=True)
