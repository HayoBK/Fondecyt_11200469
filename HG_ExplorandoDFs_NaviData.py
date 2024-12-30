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
from scipy.stats import kruskal

Py_Processing_Dir  = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")

file = Py_Processing_Dir + "C2_SimianMaze_Z3_Resumen_Short_df.csv"
main_df = pd.read_csv(file)

# Filtrar por modalidad "No Inmersivo"
main_df_filtered = main_df[main_df['Modalidad'] == "No Inmersivo"].copy()

# Seleccionar solo las columnas numéricas relevantes para el análisis
numeric_cols = main_df_filtered.select_dtypes(include=['number']).columns
columns_to_keep = ['Sujeto', 'Grupo', 'True_Block']
numeric_cols = [col for col in numeric_cols if col not in columns_to_keep]

# Promediar los True_Trial para cada Sujeto y True_Block
averaged_df = (main_df_filtered.groupby(columns_to_keep)[numeric_cols]
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
    for block in ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']:
        block_df = filtered_df[filtered_df['True_Block'] == block]

        # Crear una tabla pivotada para ANOVA
        pivot_df = block_df.pivot_table(index='Sujeto',
                                         columns='Grupo',
                                         values=var,
                                         aggfunc='mean')

        # Filtrar valores no nulos
        #pivot_df = pivot_df.dropna(axis=0, how='any')
        pivot_df.reset_index(drop=True, inplace=True)
        #pivot_df = pivot_df.drop(columns=['Grupo'])

        print(pivot_df.columns)

        if not pivot_df.empty and pivot_df.shape[1] > 1:
            # ANOVA
            anova_results = f_oneway(*[pivot_df[col] for col in pivot_df.columns])

            # Calcular tamaños de efecto (eta squared)
            eta_squared = pg.anova(data=block_df, dv=var, between='Grupo')['np2'].iloc[0]

            # Guardar resultados
            results.append({
                'Variable': var,
                'True_Block': block,
                'F-Value': anova_results.statistic,
                'p-Value': anova_results.pvalue,
                'Eta-Squared': eta_squared
            })


# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Ordenar por tamaño de efecto
#results_df = results_df.sort_values(by=['Variable', 'True_Block', 'Eta-Squared'], ascending=[True, True, False])

# Mostrar resultados
print(results_df)
print("Work's Done!")
