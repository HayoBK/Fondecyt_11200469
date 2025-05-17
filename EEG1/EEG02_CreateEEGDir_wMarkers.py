#%%

import HA_ModuloArchivos as H_Mod
import os
import sys
import mne
import pandas as pd
print("Current working directory:", os.getcwd())
eeg1_dir = os.path.abspath('./EEG1')  # O pon ruta absoluta manual

print("Intentando agregar:", eeg1_dir)
if eeg1_dir not in sys.path:
    sys.path.append(eeg1_dir)

import EEG00_HMod

# Inicialización de rutas
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")
output_dir = os.path.join(Py_Processing_Dir, "EEG_wMarkers")
os.makedirs(output_dir, exist_ok=True)

# Obtener todos los códigos de sujeto
lista_sujetos = [f for f in os.listdir(Py_Sujetos_Dir) if
                 f.startswith('P') and os.path.isdir(os.path.join(Py_Sujetos_Dir, f))]

log_resultados = []

for Sujeto in sorted(lista_sujetos):
    try:
        print(f"\nProcesando {Sujeto}...")
        Py_Specific_Dir = os.path.join(Py_Sujetos_Dir, Sujeto, 'EEG')
        eegfile = os.path.join(Py_Specific_Dir, f"{Sujeto}_NAVI.vhdr")

        archivos_esperados = [
            'trials_forMATLAB_NI.csv', 'fixation_forMATLAB_NI.csv', 'blinks_forMATLAB_NI.csv',
            'trials_forMATLAB_RV.csv', 'fixation_forMATLAB_RV.csv', 'blinks_forMATLAB_RV.csv'
        ]
        archivos_encontrados = all(os.path.exists(os.path.join(Py_Specific_Dir, f)) for f in archivos_esperados)

        if not os.path.exists(eegfile):
            raise FileNotFoundError(f"Archivo EEG no encontrado: {eegfile}")
        if not archivos_encontrados:
            raise FileNotFoundError(f"Faltan archivos .csv esperados en {Py_Specific_Dir}")

        EEG00_HMod.generar_Sync_Markers_Files(Sujeto, H_Mod)
        raw = mne.io.read_raw_brainvision(eegfile, preload=True)

        resultados_sync = EEG00_HMod.calcular_delta_sync(Sujeto, raw, H_Mod)
        delta_promedio_NI = resultados_sync['NI']['promedio']
        delta_promedio_RV = resultados_sync['RV']['promedio']

        raw, _ = EEG00_HMod.integrar_time_markers_en_raw(
            raw, Py_Specific_Dir,
            'trials_forMATLAB_NI.csv',
            'fixation_forMATLAB_NI.csv',
            'blinks_forMATLAB_NI.csv',
            delta_promedio=delta_promedio_NI,
            modalidad='NI',
            modo_operacion='solo'
        )

        raw, _ = EEG00_HMod.integrar_time_markers_en_raw(
            raw, Py_Specific_Dir,
            'trials_forMATLAB_RV.csv',
            'fixation_forMATLAB_RV.csv',
            'blinks_forMATLAB_RV.csv',
            delta_promedio=delta_promedio_RV,
            modalidad='RV',
            modo_operacion='add'
        )

        # Guardar archivos .fif en dos ubicaciones
        fif_name = f"{Sujeto}_conLSLMarkers_eeg.fif"
        path_local = os.path.join(Py_Specific_Dir, fif_name)
        path_global = os.path.join(output_dir, fif_name)
        raw.save(path_local, overwrite=True)
        raw.save(path_global, overwrite=True)

        log_resultados.append((Sujeto, "Procesado exitosamente"))

    except Exception as e:
        log_resultados.append((Sujeto, f"ERROR: {str(e)}"))

# Crear reporte en consola
print("\n==== REPORTE FINAL ====")
for sujeto, estado in log_resultados:
    print(f"{sujeto}: {estado}")

# Exportar log a archivo CSV si se desea
df_log = pd.DataFrame(log_resultados, columns=["Sujeto", "Estado"])
df_log.to_csv(os.path.join(output_dir, "log_procesamiento.csv"), index=False)
