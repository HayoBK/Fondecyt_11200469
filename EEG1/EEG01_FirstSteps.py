# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2025 - Mayo - 12, Lunes.
# Veamos como nos va con EEG MNE.
# -----------------------------------------------------------------------

# Aprendizajes
# 1) MNE tiene una forma de cargar nativamente  BrainVision files de los que usa Billeke
# 2) MNE necesita además de cargar MNE, cargar la biblioteca PyQt5

# Este script es una mierda desordenada para poder aprender a manejar MNE y analizar los problemas de sincronización entre EEG y LSL

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

LSLmarkers, LSLdata = EEG00_HMod.generar_Sync_Markers_Filesb(Sujeto,H_Mod)  # Extramos de LSL los marcadores de Eventos
LSLmarkers = LSLmarkers[LSLmarkers['OverWatch_MarkerA'] != 'NONE']

Py_Specific_Dir = Py_Sujetos_Dir + Sujeto + '/EEG/'
eegfile = Py_Specific_Dir + Sujeto + '_NAVI.vhdr'  # Cargamos el EEG File crudo

# Cargar los datos
raw = mne.io.read_raw_brainvision(eegfile, preload=True)

#Intentemos aqui trbajar con las anotaciones
annotations = raw.annotations
ori_ann_df, new_ann_df = EEG00_HMod.traducir_anotaciones_originales_EEG(annotations)
EEGMarkers = new_ann_df.rename(columns={
    'onset': 'OverWatch_time_stamp',
    'description': 'OverWatch_MarkerA'
})[['OverWatch_time_stamp', 'OverWatch_MarkerA']]
EEGMarkers['OverWatch_MarkerA'] = EEGMarkers['OverWatch_MarkerA'].astype(str)

exito, output = H_Mod.ClearMarkers_LEGACY(EEGMarkers)
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


print (raw.info['meas_date'].timestamp())

EEGMarkers_filtrada = EEG00_HMod.filtrar_marcadores(EEGMarkers, 'EEG')
LSLmarkers_filtrada = EEG00_HMod.filtrar_marcadores(LSLmarkers, 'LSL')

EEGMarkers_filtrada = EEGMarkers_filtrada.reset_index(drop=True)
LSLmarkers_filtrada = LSLmarkers_filtrada.reset_index(drop=True)

# Agrupar por tipo de marcador para comparar por tipo
tipos_comunes = set(EEGMarkers_filtrada['OverWatch_MarkerA']).intersection(
                 set(LSLmarkers_filtrada['OverWatch_MarkerA']))

def orden_marcadores(x):
    try:
        return (0, int(x))  # los numéricos primero
    except ValueError:
        return (1, x)        # luego los strings como "Stop"


for tipo in sorted(tipos_comunes, key=orden_marcadores):
    eeg_times = EEGMarkers_filtrada.query("OverWatch_MarkerA == @tipo")['OverWatch_time_stamp'].reset_index(drop=True)
    lsl_times = LSLmarkers_filtrada.query("OverWatch_MarkerA == @tipo")['OverWatch_time_stamp'].reset_index(drop=True)

    n = min(len(eeg_times), len(lsl_times))
    print(f"\n🔍 Comparando marcador: {tipo} (primeros {n} eventos)")

    for i in range(n):
        delta = eeg_times[i] - lsl_times[i]
        print(f"  Evento {i + 1}: EEG = {eeg_times[i]:.3f}  |  LSL = {lsl_times[i]:.3f}  |  Δ = {delta:.3f} s")

    if len(eeg_times) != len(lsl_times):
        print(f"⚠️ Diferente número de eventos para '{tipo}': EEG = {len(eeg_times)} vs LSL = {len(lsl_times)}")

EEGTrials_limpios = EEG00_HMod.limpiar_trials_eeg(EEGMarkers)


#%% Integrar los eventos de modalidad NI
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
