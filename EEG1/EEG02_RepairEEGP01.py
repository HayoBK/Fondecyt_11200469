import os
import mne
import numpy as np
import HA_ModuloArchivos as H_Mod
from datetime import timedelta

# === Definir rutas ===
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")
sujeto = "P01"
dir_sujeto = os.path.join(Py_Sujetos_Dir, sujeto, "EEG")

# === Archivos segmentados ===
vhdr_files = [
    os.path.join(dir_sujeto, "navi_p01.vhdr"),
    os.path.join(dir_sujeto, "navirv_p01.vhdr"),
    os.path.join(dir_sujeto, "navirv2_p01.vhdr")
]

# === Cargar archivos como RawBrainVision ===
print(f"Cargando archivos para {sujeto}...")
raws = []
for path in vhdr_files:
    if os.path.exists(path):
        raw = mne.io.read_raw_brainvision(path, preload=True)
        raws.append(raw)
    else:
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

# === Extraer fechas de inicio y duraciones ===
start_times = [r.info['meas_date'] for r in raws]
duraciones = [r.times[-1] for r in raws]  # en segundos
sfreq = raws[0].info['sfreq']
n_channels = raws[0].info['nchan']

# === Convertir a RawArray para manipulación homogénea ===
raw_arrays = []
for raw in raws:
    data, times = raw.get_data(return_times=True)
    raw_array = mne.io.RawArray(data, raw.info)
    raw_arrays.append(raw_array)

# === Concatenar con lagunas según tiempo real ===
raw_combined = raw_arrays[0]
for i in range(1, len(raw_arrays)):
    delta_real = (start_times[i] - start_times[i - 1]).total_seconds()
    delta_real -= duraciones[i - 1]

    if delta_real <= 0:
        print(f"⚠️ Tiempo negativo o nulo entre segmentos {i-1} y {i}. Insertando laguna mínima de 1s.")
        delta_real = 1.0

    print(f"⏱️ Insertando laguna de {delta_real:.2f} segundos entre segmentos {i-1} y {i}")
    n_samples_gap = int(delta_real * sfreq)
    data_gap = np.zeros((n_channels, n_samples_gap))
    raw_gap = mne.io.RawArray(data_gap, raws[i].info.copy())

    raw_combined.append([raw_gap, raw_arrays[i]])  # no reasignar

# === Guardar archivo final ===
fif_name = f"{sujeto}_con_lagunas_cronologicas.fif"
path_local = os.path.join(dir_sujeto, fif_name)
raw_combined.save(path_local, overwrite=True)

print(f"\n✅ Archivo combinado y guardado en:\n- {path_local}")