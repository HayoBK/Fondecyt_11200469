# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Diciembre - 28, Sábado.
# Script para procesar Los DF's que ya llevo, pensando en
# los otros valores de Navegación más allá del CSE
# y en al artículo para Rosario
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd


Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

#EJEMPLO DE USO de Explorar_DF .... con esto obtuvimos la descripción de los archivos .csv
dataframes = H_Mod.Explorar_DF(Py_Processing_Dir)
# A_INFINITE_BASAL_DF.xlsx
# C2_SimianMaze_Z3_Resumen_Short_df
print(" Work's Done! ")