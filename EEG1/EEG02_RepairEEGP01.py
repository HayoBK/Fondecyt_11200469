
import os
import mne
import numpy as np
import HA_ModuloArchivos as H_Mod
# Definir rutas
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

# === Convertir los RawBrainVision a RawArray (mantiene tipo homogéneo) ===
raw_arrays = []
for raw in raws:
    data, times = raw.get_data(return_times=True)
    raw_array = mne.io.RawArray(data, raw.info)
    raw_arrays.append(raw_array)

# === Crear laguna de 5 segundos ===
sfreq = raw_arrays[0].info['sfreq']
n_channels = raw_arrays[0].info['nchan']
n_samples_gap = int(5 * sfreq)
data_gap = np.zeros((n_channels, n_samples_gap))
raw_gap = mne.io.RawArray(data_gap, raw_arrays[0].info.copy())

# === Concatenar con lagunas ===
raw_combined = raw_arrays[0]
for r in raw_arrays[1:]:
    raw_combined.append([raw_gap, r])  # NO asignes, solo llama al método

# === Guardar en .fif ===
fif_name = f"{sujeto}_con_lagunas.fif"
path_local = os.path.join(dir_sujeto, fif_name)
raw_combined.save(path_local, overwrite=True)

print(f"✅ Archivo guardado en:\n- {path_local}")

print(f"✅ Archivo guardado en:\n- {path_local}")