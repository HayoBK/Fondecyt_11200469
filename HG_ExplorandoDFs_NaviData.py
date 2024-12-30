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
main_df = main_df[main_df['Sujeto'] != 'P13']
main_df = main_df[main_df['Sujeto'] != 'P37']
main_df = main_df[main_df['Sujeto'] != 'P44']
# Filtrar por modalidad "No Inmersivo"
#main_df_filtered = main_df[main_df['Modalidad'] == "No Inmersivo"].copy()

# Seleccionar solo las columnas numéricas relevantes para el análisis
numeric_cols = main_df.select_dtypes(include=['number']).columns
columns_to_keep = ['Sujeto', 'Grupo','Modalidad', 'True_Block']
numeric_cols = [col for col in numeric_cols if col not in columns_to_keep]

# Promediar los True_Trial para cada Sujeto y True_Block
averaged_df = (main_df.groupby(columns_to_keep)[numeric_cols]
               .mean()
               .reset_index())

# Filtrar los True_Block relevantes
filtered_df = averaged_df[averaged_df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3'])]

# Variables de interés
output_vars = ["Entropia_Espacial", "Herror", "Hpath", "Htotal", "Indice_Eficiencia", "Latencia", "Velocidad_Promedio", "CSE"]

# Resultados para tamaños de efecto y estadísticas
results = []
modalidades =['No Inmersivo', 'Realidad Virtual']
# Analizar cada variable output

results = {
    'Mod': [],
    'Block': [],
    'Var': [],
    'P-Value': [],
    'Effect Size (Eta-Squared)': []
}
for var in output_vars:
    for Mod in modalidades:
        for block in ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']:
            Mod_df = filtered_df[filtered_df['Modalidad']== Mod]
            df = Mod_df[Mod_df['True_Block'] == block]
            print(Mod, block, var)
            group_means = df.groupby('Grupo')[var].mean()
            print("Promedios por grupo:")
            print(group_means)

            # Realizar ANOVA
            groups = [df[df['Grupo'] == g][var].dropna().values for g in df['Grupo'].unique()]
            anova_results = f_oneway(*groups)

            # Calcular el tamaño del efecto (eta cuadrado parcial)
            anova_df = pg.anova(data=df, dv=var, between='Grupo')

            print("\nResultados del ANOVA:")
            print(f"F-Value: {anova_results.statistic}")
            print(f"P-Value: {anova_results.pvalue}")

            print("\nTamaño del efecto (eta cuadrado parcial):")
            print(anova_df['np2'][0])

            results['Mod'].append(Mod)
            results['Block'].append(block)
            results['Var'].append(var)
            results['P-Value'].append(anova_results.pvalue)
            results['Effect Size (Eta-Squared)'].append(anova_df['np2'][0])
results_df = pd.DataFrame(results)


print("Work's Done!")
