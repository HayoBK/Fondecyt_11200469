
import os
import mne
import numpy as np
import HA_ModuloArchivos as H_Mod
# Definir rutas
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

sujeto = "P01"
dir_sujeto = os.path.join(Py_Sujetos_Dir, sujeto, "EEG")

# === CARGAR LOS SEGMENTOS ===
vhdr_files = [
    os.path.join(dir_sujeto, f"{sujeto}_parte1.vhdr"),
    os.path.join(dir_sujeto, f"{sujeto}_parte2.vhdr"),
    os.path.join(dir_sujeto, f"{sujeto}_parte3.vhdr")
]

print(f"Cargando archivos para {sujeto}...")
raws = []
for path in vhdr_files:
    if os.path.exists(path):
        raw = mne.io.read_raw_brainvision(path, preload=True)
        raws.append(raw)
    else:
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

# === CREAR LAGUNAS DE 5 SEGUNDOS ENTRE SEGMENTOS ===
sfreq = raws[0].info['sfreq']
n_channels = raws[0].info['nchan']
duration_gap = 5  # segundos
n_samples_gap = int(duration_gap * sfreq)
data_gap = np.zeros((n_channels, n_samples_gap))
raw_gap = mne.io.RawArray(data_gap, raws[0].info.copy())

# === CONCATENAR TODOS LOS TRAMOS CON LAGUNAS INTERMEDIAS ===
raw_combined = raws[0]
for r in raws[1:]:
    raw_combined = raw_combined.append([raw_gap, r])

# === GUARDAR EL ARCHIVO FINAL .FIF ===
fif_name = f"{sujeto}_con_lagunas.fif"
path_local = os.path.join(dir_sujeto, fif_name)
path_global = os.path.join(output_dir, fif_name)

print("Guardando archivo unificado...")
raw_combined.save(path_local, overwrite=True)
raw_combined.save(path_global, overwrite=True)

print(f"✅ Archivo guardado en:\n- {path_local}\n- {path_global}")