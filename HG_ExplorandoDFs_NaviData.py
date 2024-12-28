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
import pingouin as pg
from scipy.stats import f_oneway

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

# A_INFINITE_BASAL_DF.xlsx
# C2_SimianMaze_Z3_Resumen_Short_df
file = Py_Processing_Dir + "C2_SimianMaze_Z3_Resumen_Short_df.csv"
main_df = pd.read_csv(file)


# Filtrar por modalidad "No Inmersivo"
main_df_filtered = main_df[main_df['Modalidad'] == "No Inmersivo"].copy()

# Promediar los True_Trial para cada Sujeto y True_Block
averaged_df = (main_df_filtered
               .groupby(['Sujeto', 'Grupo', 'True_Block'])
               .mean()
               .reset_index())

# Filtrar los True_Block relevantes
filtered_df = averaged_df[averaged_df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]

# Variables de interés
output_vars = ["Entropia_Espacial", "Herror", "Hpath", "Htotal", "Indice_Eficiencia", "Latencia", "Velocidad_Promedio", "CSE"]

# Resultados para tamaños de efecto y estadísticas
results = []

# Analizar cada variable output
for var in output_vars:
    # Crear una tabla pivotada para ANOVA
    pivot_df = filtered_df.pivot_table(index='Sujeto',
                                       columns='Grupo',
                                       values=var,
                                       aggfunc='mean')

    # Filtrar valores no nulos
    pivot_df = pivot_df.dropna(axis=0, how='any')

    # ANOVA
    anova_results = f_oneway(*[pivot_df[col] for col in pivot_df.columns])

    # Calcular tamaños de efecto (eta squared)
    eta_squared = pg.anova(data=filtered_df, dv=var, between='Grupo')['np2'][0]

    # Guardar resultados
    results.append({
        'Variable': var,
        'F-Value': anova_results.statistic,
        'p-Value': anova_results.pvalue,
        'Eta-Squared': eta_squared
    })

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Ordenar por tamaño de efecto
results_df = results_df.sort_values(by='Eta-Squared', ascending=False)

# Mostrar resultados
print(results_df)


print(" Work's Done! ")