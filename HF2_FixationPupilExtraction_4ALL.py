# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 20, Lunes.
# Script para procesar de Pupil Labs usando Lab Recorder como indice
# Pero incorporando datos de Fijaciones para TODOS los SUJETOS.
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# Lista de sujetos (excluyendo P09)
sujetos = [f"P{i:02}" for i in range(1, 54) if i not in [3, 9]]
grand_df = pd.DataFrame()
error_report = []

for Sujeto in sujetos:
    print(f"Procesando sujeto: {Sujeto}")
    Dir = Sujetos_Dir + f"{Sujeto}/"

    # Procesar modalidad "NI"
    print("\nProcesando modalidad NI")
    try:
        Reporte, file = H_Mod.Grab_LabRecorderFile("NI", Dir)
        print(Reporte)

        file_path = Sujetos_Dir + f"{Sujeto}/PUPIL_LAB/000/exports/000/surfaces/fixations_on_surface_Hefestp 1.csv"
        fixations_df = pd.read_csv(file_path)
        Interesting_df = pd.read_csv(file_path)

        file_path = Sujetos_Dir + f"{Sujeto}/PUPIL_LAB/000/exports/000/blinks.csv"
        blinks_df = pd.read_csv(file_path)

        processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels = H_Mod.process_xdf_files(file)
        print('Checking: Se procesaron una Cantidad de archivo XDF igual a (ojalá 1): ', processed_files)

        archivo = Sujetos_Dir + f"{Sujeto}/EEG/export_for_MATLAB_Sync_NI.csv"
        sync_df.to_csv(archivo, index=False)

        Interesting_df = H_Mod.Binnear_DF(trial_labels, trials_per_trialLabel, Interesting_df, 'start_timestamp')
        # Eliminar filas donde OW_Trial sea NaN
        Interesting_df.dropna(subset=['OW_Trial'], inplace=True)

        # Agregar columnas "Sujeto" y "Modalidad"
        Interesting_df.insert(0, 'Sujeto', Sujeto)
        Interesting_df.insert(1, 'Modalidad', 'No Inmersivo')

        # Combinar en el DataFrame principal
        grand_df = pd.concat([grand_df, Interesting_df], ignore_index=True)

        output_df = blinks_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        output_df = output_df[['start_time', 'duration']]
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/blinks_forMATLAB_NI.csv"
        output_df.to_csv(archivo, index=False)

        output_df = fixations_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        output_df = output_df.drop_duplicates(subset='fixation_id', keep='first')
        output_df = output_df[['start_time', 'duration']]
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/fixation_forMATLAB_NI.csv"
        output_df.to_csv(archivo, index=False)

        output_df = trials_per_trialLabel.rename(columns={
            'OW_trials': 'trial_id',
            'Start': 'start_time',
            'End': 'end_time'
        })
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/trials_forMATLAB_NI.csv"
        output_df.to_csv(archivo, index=False)

    except Exception as e:
        error_report.append(f"Error procesando NI para {Sujeto}: {e}")

    # Procesar modalidad "RV"
    print("\nProcesando modalidad RV")
    try:
        Reporte, file = H_Mod.Grab_LabRecorderFile("RV", Dir)
        print(Reporte)

        file_path = Sujetos_Dir + f"{Sujeto}/PUPIL_LAB/001/exports/000/fixations.csv"
        fixations_df = pd.read_csv(file_path)
        Interesting_df = pd.read_csv(file_path)

        file_path = Sujetos_Dir + f"{Sujeto}/PUPIL_LAB/001/exports/000/blinks.csv"
        blinks_df = pd.read_csv(file_path)

        processed_files, sync_df, trials_per_timestamp, trials_per_trialLabel, trial_labels = H_Mod.process_xdf_files(file)
        print('Checking: Se procesaron una Cantidad de archivo XDF igual a (ojalá 1): ', processed_files)

        archivo = Sujetos_Dir + f"{Sujeto}/EEG/export_for_MATLAB_Sync_RV.csv"
        sync_df.to_csv(archivo, index=False)

        Interesting_df = H_Mod.Binnear_DF(trial_labels, trials_per_trialLabel, Interesting_df, 'start_timestamp')
        # Eliminar filas donde OW_Trial sea NaN
        Interesting_df.dropna(subset=['OW_Trial'], inplace=True)

        # Agregar columnas "Sujeto" y "Modalidad"
        Interesting_df.insert(0, 'Sujeto', Sujeto)
        Interesting_df.insert(1, 'Modalidad', 'Realidad Virtual')

        # Combinar en el DataFrame principal
        grand_df = pd.concat([grand_df, Interesting_df], ignore_index=True)


        output_df = blinks_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        output_df = output_df[['start_time', 'duration']]
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/blinks_forMATLAB_RV.csv"
        output_df.to_csv(archivo, index=False)

        output_df = fixations_df.rename(columns={
            'start_timestamp': 'start_time',
            'duration': 'duration'
        })
        output_df = output_df.drop_duplicates(subset='id', keep='first')
        output_df = output_df[['start_time', 'duration']]
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/fixation_forMATLAB_RV.csv"
        output_df.to_csv(archivo, index=False)

        output_df = trials_per_trialLabel.rename(columns={
            'OW_trials': 'trial_id',
            'Start': 'start_time',
            'End': 'end_time'
        })
        archivo = Sujetos_Dir + f"{Sujeto}/EEG/trials_forMATLAB_RV.csv"
        output_df.to_csv(archivo, index=False)

    except Exception as e:
        error_report.append(f"Error procesando RV para {Sujeto}: {e}")

# Exportar reporte de errores
with open(Py_Processing_Dir + "F_FixationError_Reports.txt", "w") as report_file:
    report_file.write("\n".join(error_report))

# Exportar DataFrame consolidado

codex_path = Py_Processing_Dir + "A_OverWatch_Codex.xlsx"
codex_df = pd.read_excel(codex_path)

grand_df = grand_df.merge(codex_df, how="left", left_on="OW_Trial", right_on="OverWatch_T")
grand_df.rename(columns={"MWM_Bloque": "True_Block", "MWM_Trial": "True_Trial"}, inplace=True)
grand_df.drop(columns=["OverWatch_T"], inplace=True)
grand_df.to_csv(Py_Processing_Dir + "F_Fixations_4All.csv", index=False)

print("Work's Done for all subjects")
#%%