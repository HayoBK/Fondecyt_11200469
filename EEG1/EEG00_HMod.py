import pandas as pd
import numpy as np
import os
import mne

def generar_Sync_Markers_Files(sujeto, H_Mod):
    # Directorios base
    Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
    Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")
    Dir = Sujetos_Dir + sujeto + "/"

    # ------------------- Modalidades: NI y RV -------------------
    for modalidad, carpeta_idx in zip(["NI", "RV"], ["000", "001"]):
        print(f"Procesando sujeto {sujeto}, modalidad {modalidad}...")

        # Buscar archivo XDF
        Reporte, file = H_Mod.Grab_LabRecorderFile(modalidad, Dir)
        print(Reporte)

        # Procesar archivo XDF
        processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels = H_Mod.process_xdf_files(file)
        print(f'Se procesaron archivos XDF: {processed_files}')

        # Guardar sync_df
        archivo_sync = f"{Sujetos_Dir}{sujeto}/EEG/export_for_MATLAB_Sync_{modalidad}.csv"
        sync_df.to_csv(archivo_sync, index=False)

        # Cargar archivos de Fixations y Blinks
        base_path = f"{Sujetos_Dir}{sujeto}/PUPIL_LAB/{carpeta_idx}/exports/000/"
        fix_file = base_path + ("surfaces/fixations_on_surface_Hefestp 1.csv" if modalidad == "NI" else "fixations.csv")
        blink_file = base_path + "blinks.csv"

        fixations_df = pd.read_csv(fix_file)
        blinks_df = pd.read_csv(blink_file)
        Interesting_df = pd.read_csv(fix_file)

        # Asignar OW_Trial a los timestamps
        Interesting_df = H_Mod.Binnear_DF(trial_labels, trials_per_trialLabel, Interesting_df, 'start_timestamp')

        # Guardar Blinks
        output_df_blinks = blinks_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        output_df_blinks = output_df_blinks[['start_time', 'duration']]
        archivo_blinks = f"{Sujetos_Dir}{sujeto}/EEG/blinks_forMATLAB_{modalidad}.csv"
        output_df_blinks.to_csv(archivo_blinks, index=False)

        # Guardar Fixations
        output_df_fix = fixations_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        id_col = 'fixation_id' if modalidad == 'NI' else 'id'
        output_df_fix = output_df_fix.drop_duplicates(subset=id_col, keep='first')
        output_df_fix = output_df_fix[['start_time', 'duration']]
        archivo_fix = f"{Sujetos_Dir}{sujeto}/EEG/fixation_forMATLAB_{modalidad}.csv"
        output_df_fix.to_csv(archivo_fix, index=False)

        # Guardar Trials
        output_df_trials = trials_per_trialLabel.rename(columns={
            'OW_trials': 'trial_id',
            'Start': 'start_time',
            'End': 'end_time'
        })
        archivo_trials = f"{Sujetos_Dir}{sujeto}/EEG/trials_forMATLAB_{modalidad}.csv"
        output_df_trials.to_csv(archivo_trials, index=False)

    print('--------------------------------------------------------------------------------------------------- ')
    print(f"\u2713 Extraccion de LSL y generación de Archivos para sincronización completo para {sujeto}.")
    print('--------------------------------------------------------------------------------------------------- ')

def calcular_delta_sync(sujeto, raw, H_Mod):
    """
    Calcula la sincronización entre eventos anotados en el EEG (MNE Raw) y los exportados desde LSL en .csv,
    para ambas modalidades: NI y RV.

    Requiere que los archivos export_for_MATLAB_Sync_[NI|RV].csv ya estén generados.

    Retorna un diccionario con las estadísticas de sincronización por modalidad.
    """
    Py_Specific_Dir = H_Mod.Nombrar_HomePath(f"002-LUCIEN/SUJETOS/{sujeto}/EEG/")
    eegfile = Py_Specific_Dir + f"{sujeto}_NAVI.vhdr"
    #raw = mne.io.read_raw_brainvision(eegfile, preload=True)
    sfreq = raw.info['sfreq']

    # Extraer anotaciones y construir estructura de eventos
    eventos = [(desc, onset * sfreq) for desc, onset in zip(raw.annotations.description, raw.annotations.onset)]

    # Equivalencias entre etiquetas
    equivalencias = {'Stimulus/S101': 1, 'Stimulus/S110': 10, 'Stimulus/S120': 20, 'Stimulus/S130': 30}

    resultados = {}

    for modalidad in ['NI', 'RV']:
        archivo_sync = os.path.join(Py_Specific_Dir, f"export_for_MATLAB_Sync_{modalidad}.csv")
        if not os.path.isfile(archivo_sync):
            print(f"[!] Archivo no encontrado para {modalidad}: {archivo_sync}")
            continue

        df_lsl = pd.read_csv(archivo_sync)
        if not {'labelLSL', 'latencyLSL'}.issubset(df_lsl.columns):
            print(f"[!] Columnas faltantes en archivo {archivo_sync}")
            continue

        labels_lsl = df_lsl['labelLSL'].astype(int).values
        latencies_lsl = df_lsl['latencyLSL'].values


        # Buscar en las anotaciones del EEG la instancia correcta (NI = 1ra, RV = 2da)
        instancias_validas = []
        for etiqueta_lan, valor_lsl in equivalencias.items():
            apariciones = [onset for desc, onset in eventos if desc == etiqueta_lan]
            if len(apariciones) == 2:
                index = 0 if modalidad == 'NI' else 1
                instancias_validas.append((valor_lsl, apariciones[index] / sfreq))

        deltas = []
        for label_lsl, latency_lan in instancias_validas:
            label_lsl = int(label_lsl)  # asegurar tipo entero
            indices = np.where(labels_lsl == label_lsl)[0]
            if len(indices) > 0:
                delta = latency_lan - latencies_lsl[indices[0]]
                deltas.append(delta)

        if deltas:
            deltas = np.array(deltas)
            delta_prom = deltas.mean()
            delta_std = deltas.std()
            delta_max = deltas.max() - deltas.min()

            tiempo_transcurrido = max([onset for _, onset in eventos]) / sfreq - min([onset for _, onset in eventos]) / sfreq
            drift_acumulado = delta_max * 1000
            drift_pct = (drift_acumulado / (tiempo_transcurrido * 1000)) * 100

            resultados[modalidad] = {
                'promedio': delta_prom,
                'std': delta_std,
                'max': delta_max,
                'drift_ms': drift_acumulado,
                'drift_pct': drift_pct
            }

            print(f"\n[✓] Resultados de sincronización para sujeto {sujeto} - modalidad {modalidad}:")
            print(f"    ▸ Δ promedio     : {delta_prom:.6f} segundos")
            print(f"    ▸ Desviación std : {delta_std:.6f} segundos")
            print(f"    ▸ Δ máximo       : {delta_max:.6f} segundos")
            print(f"    ▸ Drift acumulado: {drift_acumulado:.2f} ms")
            print(f"    ▸ Porcentaje drift: {drift_pct:.4f}% del tiempo total")
        else:
            print(f"[!] No se encontraron deltas para {modalidad} en {sujeto}.")

    return resultados


import pandas as pd
import numpy as np
import os
import mne


def integrar_time_markers_en_raw(raw, ruta, archivo_trials, archivo_fijaciones, archivo_blinks, delta_promedio,
                                 modalidad, modo_operacion='solo'):
    """
    Incorpora eventos de trials, fijaciones y blinks a un objeto MNE Raw, ajustando latencias según el delta promedio.

    Devuelve el objeto Raw modificado y la lista de IDs de trials encontrados.
    """
    # Cargar archivos CSV
    trials = pd.read_csv(os.path.join(ruta, archivo_trials))
    fijaciones = pd.read_csv(os.path.join(ruta, archivo_fijaciones))
    blinks = pd.read_csv(os.path.join(ruta, archivo_blinks))

    # Ajustar latencias (en segundos)
    trials['start_corrected'] = trials['start_time'] + delta_promedio
    trials['end_corrected'] = trials['end_time'] + delta_promedio
    trials['duration'] = trials['end_corrected'] - trials['start_corrected']

    fijaciones['start_corrected'] = fijaciones['start_time'] + delta_promedio
    fijaciones['end_corrected'] = fijaciones['start_corrected'] + (fijaciones['duration'] / 1000)

    blinks['start_corrected'] = blinks['start_time'] + delta_promedio
    blinks['end_corrected'] = blinks['start_corrected'] + blinks['duration']

    # Filtrar eventos dentro del rango de los trials
    inicio_trials = trials['start_corrected'].min()
    fin_trials = trials['end_corrected'].max()

    fijaciones = fijaciones[
        (fijaciones['start_corrected'] >= inicio_trials) & (fijaciones['end_corrected'] <= fin_trials)]
    blinks = blinks[(blinks['start_corrected'] >= inicio_trials) & (blinks['end_corrected'] <= fin_trials)]

    # Crear anotaciones
    anotaciones = []

    # Etiquetas enriquecidas de MWM
    MWM_labels_base = [
        'T01_FreeNav', 'T02_Training',
        'T03_NaviVT1_i1', 'T04_NaviVT1_i2', 'T05_NaviVT1_i3', 'T06_NaviVT1_i4',
        'T07_NaviHT1_i1', 'T08_NaviHT1_i2', 'T09_NaviHT1_i3', 'T10_NaviHT1_i4',
        'T11_NaviHT1_i5', 'T12_NaviHT1_i6', 'T13_NaviHT1_i7', 'T14_Rest1',
        'T15_NaviHT2_i1', 'T16_NaviHT2_i2', 'T17_NaviHT2_i3', 'T18_NaviVT2_i4',
        'T19_NaviHT2_i5', 'T20_NaviHT2_i6', 'T21_NaviHT2_i7', 'T22_Rest2',
        'T23_NaviHT3_i1', 'T24_NaviHT3_i2', 'T25_NaviHT3_i3', 'T26_NaviHT3_i4',
        'T27_NaviHT3_i5', 'T28_NaviHT3_i6', 'T29_NaviHT3_i7', 'T30_Rest3',
        'T31_NaviVT2_i1', 'T32_NaviVT2_i2', 'T33_NaviVT2_i3']

    MWM_labels = [f"{modalidad}_{label}" for label in MWM_labels_base]
    label_map = {i + 1: MWM_labels[i] for i in range(len(MWM_labels))}

    # Agregar anotaciones de trials
    for _, row in trials.iterrows():
        label = label_map.get(row['trial_id'], f'TRIAL_{int(row["trial_id"])}')
        anotaciones.append((row['start_corrected'], row['duration'], label))

    # Agregar fijaciones
    for _, row in fijaciones.iterrows():
        dur = row['end_corrected'] - row['start_corrected']
        anotaciones.append((row['start_corrected'], dur, 'FIXATION'))

    # Agregar blinks
    for _, row in blinks.iterrows():
        dur = row['end_corrected'] - row['start_corrected']
        anotaciones.append((row['start_corrected'], dur, 'BLINK'))

    # Añadir o reemplazar anotaciones
    new_annotations = mne.Annotations(
        onset=[a[0] for a in anotaciones],
        duration=[a[1] for a in anotaciones],
        description=[a[2] for a in anotaciones],
        orig_time=raw.annotations.orig_time
    )

    if modo_operacion == 'add':
        raw.set_annotations(raw.annotations + new_annotations)
    else:
        raw.set_annotations(new_annotations)

    # Guardar reporte de trials
    unique_trials = sorted(trials['trial_id'].unique())
    expected_trials = list(range(1, max(unique_trials) + 1))
    missing_trials = list(set(expected_trials) - set(unique_trials))

    reporte_path = os.path.join(ruta, f'Reporte_Trials_{modalidad}.txt')
    with open(reporte_path, 'w') as f:
        f.write(f"Trials encontrados: {unique_trials}\n")
        if missing_trials:
            f.write(f"Trials faltantes: {missing_trials}\n")
        else:
            f.write("No hay trials faltantes.\n")

    print(f"[✓] Anotaciones integradas en modalidad {modalidad}. Reporte guardado en {reporte_path}")
    return raw, unique_trials