# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2025 - Enero - 17, .
# Script para identificar Outliers
# Además me fui en volada y me puse a comparar si en efecto habia diferencias entre Alo y Ego centrico.
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import os
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, zscore


Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/HI_CompAloEgo/")
file = os.path.join(Py_Processing_Dir, "H_SimianMaze_ShortDf_NoOutliers.csv")
df = pd.read_csv(file)

output_vars = ["Duration(ms)", "Path_length", "Entropia_Espacial", "Herror", "Hpath", "Htotal", "Indice_Eficiencia", "Latencia", "Velocidad_Promedio", "CSE"]
Bloques_Interesantes = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
modalidades = ['No Inmersivo', 'Realidad Virtual']
column_names = ['Sujeto', 'Grupo', 'Modalidad', 'True_Block', 'True_Trial', 'Trial_Unique_ID', 'Duration(ms)', 'Path_length', 'CSE', 'Platform_exists', 'Edad', 'Genero', 'Dg', 'Latencia', 'Velocidad_Promedio', 'Freezing', 'Indice_Eficiencia', 'Entropia_Espacial', 'Estrategia_Simple', 'Herror', 'Hpath', 'Htotal', 'Estrategia']

# Filtrar para incluir solo los bloques interesantes y modalidades
print("Filtrando DataFrame para bloques y modalidades interesantes...")
df_filtered = df[df["True_Block"].isin(Bloques_Interesantes) & df["Modalidad"].isin(modalidades)].copy()

# Normalización dentro de cada combinación de modalidad y bloque
print("Normalizando las variables de salida por modalidad y bloque...")
for var in output_vars:
    df_filtered[f"{var}_norm"] = df_filtered.groupby(["Modalidad", "True_Block"])[var].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

# Guardar el DataFrame normalizado para análisis posterior
output_file = os.path.join(Py_Processing_Dir, "H_SimianMaze_ShortDf_Normalized.csv")
df_filtered.to_csv(output_file, index=False)

print(f"DataFrame filtrado y normalizado guardado en: {output_file}")

# Evaluar diferencias entre grupos por bloque y modalidad
print("Evaluando diferencias entre grupos...")
report = []

for modalidad in modalidades:
    for bloque in Bloques_Interesantes:
        df_contexto = df_filtered[(df_filtered["Modalidad"] == modalidad) & (df_filtered["True_Block"] == bloque)]

        for var in output_vars + [f"{var}_norm" for var in output_vars]:
            grupos = [df_contexto[df_contexto["Grupo"] == grupo][var].dropna() for grupo in
                      df_contexto["Grupo"].unique()]

            if len(grupos) > 1 and all(len(g) > 1 for g in grupos):
                f_stat, p_value = f_oneway(*grupos)

                # Tamaño del efecto: Eta cuadrado (eta^2)
                ss_between = sum(len(g) * (g.mean() - df_contexto[var].mean()) ** 2 for g in grupos)
                ss_total = sum((x - df_contexto[var].mean()) ** 2 for x in df_contexto[var].dropna())
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                report.append({
                    "Modalidad": modalidad,
                    "Bloque": bloque,
                    "Variable": var,
                    "F-Statistic": f_stat,
                    "p-Value": p_value,
                    "Eta Squared": eta_squared
                })

# Convertir el reporte en un DataFrame
report_df = pd.DataFrame(report)



# Loop para imprimir resultados en pantalla
print("\nResultados de diferencias por variable:\n")
for var in output_vars + [f"{var}_norm" for var in output_vars]:
    print(f"\nVariable: {var}")
    var_results = report_df[report_df["Variable"] == var]
    for _, row in var_results.iterrows():
        print(f"Modalidad: {row['Modalidad']}, Bloque: {row['Bloque']}, F-Statistic: {row['F-Statistic']:.2f}, p-Value: {row['p-Value']:.4f}, Eta Squared: {row['Eta Squared']:.4f}")

# Filtrar resultados significativos y ordenarlos por tamaño del efecto
significant_results = report_df[(report_df["p-Value"] < 0.05) & (report_df["Modalidad"] == "No Inmersivo")].sort_values(by="Eta Squared", ascending=False)

# Imprimir resultados significativos ordenados
print("\nResultados significativos ordenados por tamaño del efecto:\n")
for _, row in significant_results.iterrows():
    print(f"Variable: {row['Variable']}, Modalidad: {row['Modalidad']}, Bloque: {row['Bloque']}, F-Statistic: {row['F-Statistic']:.2f}, p-Value: {row['p-Value']:.4f}, Eta Squared: {row['Eta Squared']:.4f}")


# Crear gráficos de boxplot para CSE_norm y Htotal_norm
print("Generando gráficos de boxplot para CSE_norm y Htotal_norm...")
for var in ["CSE_norm", "Htotal_norm", "Hpath", "Path_length"]:
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_filtered,
        x="Grupo",
        y=var,
        hue="True_Block"
    )
    plt.title(f"Boxplot de {var} por Grupo y Bloque de Interés")
    plt.xlabel("Grupo")
    plt.ylabel(var)
    plt.legend(title="Bloque de Interés")
    plot_file = os.path.join(Output_Dir, f"Boxplot_{var}.png")
    plt.savefig(plot_file)
    plt.show()
    print(f"Gráfico guardado en: {plot_file}")

# Evaluar diferencias entre HiddenTarget_3 y el menor valor entre HiddenTarget_1 y HiddenTarget_2
print("Evaluando diferencias entre bloques para cada grupo...")
comparison_results = []

for grupo in df_filtered["Grupo"].unique():
    df_grupo = df_filtered[df_filtered["Grupo"] == grupo]
    for var in output_vars + [f"{var}_norm" for var in output_vars]:
        for modalidad in modalidades:
            df_mod = df_grupo[df_grupo["Modalidad"] == modalidad]

            # Valores por bloque
            ht1_values = df_mod[df_mod["True_Block"] == "HiddenTarget_1"][var]
            ht2_values = df_mod[df_mod["True_Block"] == "HiddenTarget_2"][var]
            ht3_values = df_mod[df_mod["True_Block"] == "HiddenTarget_3"][var]

            # Comparar HiddenTarget_3 con el menor valor entre HiddenTarget_1 y HiddenTarget_2
            min_block_values = pd.concat([ht1_values, ht2_values]).groupby(level=0).min()

            if not ht3_values.empty and not min_block_values.empty:
                t_stat, p_value = ttest_ind(ht3_values, min_block_values, nan_policy='omit')

                # Tamaño del efecto: d de Cohen
                mean_diff = ht3_values.mean() - min_block_values.mean()
                pooled_std = (((ht3_values.std() ** 2) + (min_block_values.std() ** 2)) / 2) ** 0.5
                cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0

                comparison_results.append({
                    "Grupo": grupo,
                    "Modalidad": modalidad,
                    "Variable": var,
                    "T-Statistic": t_stat,
                    "p-Value": p_value,
                    "Cohens d": cohen_d
                })

# Convertir los resultados en un DataFrame
comparison_df = pd.DataFrame(comparison_results)

# Comparar tamaños de efecto entre grupos
effect_diff_group = comparison_df.groupby("Variable").apply(
    lambda x: x.groupby("Grupo")["Cohens d"].mean().max() - x.groupby("Grupo")["Cohens d"].mean().min()
).reset_index(name="Effect Size Difference Between Groups")

# Ordenar por la mayor diferencia en tamaño del efecto entre grupos
effect_diff_sorted = effect_diff_group.sort_values(by="Effect Size Difference Between Groups", ascending=False)

# Imprimir resultados detallados
print("\nResultados detallados por variable y grupo:\n")
for _, row in comparison_df.iterrows():
    print(f"Grupo: {row['Grupo']}, Modalidad: {row['Modalidad']}, Variable: {row['Variable']}, "
          f"T-Statistic: {row['T-Statistic']:.2f}, p-Value: {row['p-Value']:.4f}, Cohen's d: {row['Cohens d']:.4f}")

# Imprimir variables ordenadas por diferencia en tamaño del efecto entre grupos
print("\nVariables ordenadas por mayor diferencia en tamaño del efecto entre grupos:\n")
print(effect_diff_sorted)

# Guardar los resultados
comparison_file = os.path.join(Output_Dir, "H_SimianMaze_ComparisonResults_ByGroup.csv")
effect_diff_file = os.path.join(Output_Dir, "H_SimianMaze_EffectSizeDifferences_ByGroup.csv")
comparison_df.to_csv(comparison_file, index=False)
effect_diff_sorted.to_csv(effect_diff_file, index=False)

print(f"\nResultados de comparación guardados en: {comparison_file}")
print(f"Diferencias en tamaño del efecto entre grupos guardadas en: {effect_diff_file}")

# Reporte para Hpath
print("\nReporte detallado para Hpath:")
hpath_results = comparison_df[comparison_df["Variable"] == "Htotal_norm"]
for _, row in hpath_results.iterrows():
    print(f"Grupo: {row['Grupo']}, Modalidad: {row['Modalidad']}, T-Statistic: {row['T-Statistic']:.2f}, "
          f"p-Value: {row['p-Value']:.4f}, Cohen's d: {row['Cohens d']:.4f}")

hpath_report_file = os.path.join(Output_Dir, "H_SimianMaze_Hpath_Report.csv")
hpath_results.to_csv(hpath_report_file, index=False)
print(f"Reporte detallado de Hpath guardado en: {hpath_report_file}")

print ('Work is Done')