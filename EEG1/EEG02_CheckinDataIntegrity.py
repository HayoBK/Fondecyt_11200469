#%%
import os
import pandas as pd
import HA_ModuloArchivos as H_Mod
import ace_tools_open as tools
# Definir rutas
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")
lista_sujetos = [f for f in os.listdir(Py_Sujetos_Dir) if f.startswith('P') and os.path.isdir(os.path.join(Py_Sujetos_Dir, f))]

# Archivos esperados
archivos_csv = [
    'trials_forMATLAB_NI.csv', 'fixation_forMATLAB_NI.csv', 'blinks_forMATLAB_NI.csv',
    'trials_forMATLAB_RV.csv', 'fixation_forMATLAB_RV.csv', 'blinks_forMATLAB_RV.csv'
]

# Inicializar lista para almacenar resultados
reporte = []

for Sujeto in sorted(lista_sujetos):
    Py_Specific_Dir = os.path.join(Py_Sujetos_Dir, Sujeto, 'EEG')
    eegfile = os.path.join(Py_Specific_Dir, f"{Sujeto}_NAVI.vhdr")

    sujeto_dict = {'Sujeto': Sujeto, 'EEG_vhdr': os.path.exists(eegfile)}

    for archivo in archivos_csv:
        path_archivo = os.path.join(Py_Specific_Dir, archivo)
        sujeto_dict[archivo] = os.path.exists(path_archivo)

    reporte.append(sujeto_dict)

# Convertir a DataFrame
df_reporte = pd.DataFrame(reporte)

# Mostrar tabla de reporte
tools.display_dataframe_to_user(name="Reporte de Archivos por Sujeto", dataframe=df_reporte)
