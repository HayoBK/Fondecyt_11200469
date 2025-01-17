# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2025 - Enero - 2, Jueves.
# Script para identificar Outliers
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Configuración inicial
Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
output_dir = os.path.join(Py_Processing_Dir, "H_Outliers")
os.makedirs(output_dir, exist_ok=True)

file = os.path.join(Py_Processing_Dir, "C2_SimianMaze_Z3_Resumen_Short_df.csv")
df = pd.read_csv(file)

# Filtrar datos
output_vars = ["CSE", "Htotal"]
blocks_of_interest = ["HiddenTarget_1", "HiddenTarget_2", "HiddenTarget_3"]
modalities = ["No Inmersivo", "Realidad Virtual"]

df_filtered = df[df["True_Block"].isin(blocks_of_interest) & df["Modalidad"].isin(modalities)]

# Calcular promedios por Grupo, Modalidad, True_Block y True_Trial
promedios_grupo = (
    df_filtered.groupby(["Grupo", "Modalidad", "True_Block", "True_Trial"])[output_vars]
    .mean()
    .reset_index()
)

# Análisis sujeto por sujeto
outliers = []  # Lista para almacenar outliers identificados
detalles_outliers = []  # Lista para almacenar los detalles de los outliers

for sujeto in df_filtered["Sujeto"].unique():
    sujeto_dir = os.path.join(output_dir, sujeto)
    os.makedirs(sujeto_dir, exist_ok=True)

    df_sujeto = df_filtered[df_filtered["Sujeto"] == sujeto]
    grupo = df_sujeto["Grupo"].iloc[0]

    for modalidad in modalities:
        for block in blocks_of_interest:
            for var in output_vars:
                var_dir = os.path.join(sujeto_dir, var)
                os.makedirs(var_dir, exist_ok=True)

                # Filtrar los datos relevantes
                df_mod_block = df_sujeto[(df_sujeto["Modalidad"] == modalidad) & (df_sujeto["True_Block"] == block)]
                promedios_mod_block = promedios_grupo[(promedios_grupo["Grupo"] == grupo) &
                                                      (promedios_grupo["Modalidad"] == modalidad) &
                                                      (promedios_grupo["True_Block"] == block)]
                if df_mod_block.empty or promedios_mod_block.empty:
                    continue

                # Crear gráfico de barras con barras de error
                plt.figure(figsize=(10, 6))
                promedios_mod_block[f"{var}_std"] = (
                    df_filtered[(df_filtered["Grupo"] == grupo) &
                                (df_filtered["Modalidad"] == modalidad) &
                                (df_filtered["True_Block"] == block) &
                                (df_filtered["True_Trial"] == promedios_mod_block["True_Trial"].iloc[0])]
                    [var]
                    .std()
                )

                df_plot = pd.concat([
                    df_mod_block[["True_Trial", var]].assign(Categoria="Sujeto"),
                    promedios_mod_block[["True_Trial", var]].assign(Categoria="Promedio Grupo")
                ])

                sns.barplot(
                    data=df_plot,
                    x="True_Trial",
                    y=var,
                    hue="Categoria",
                    ci=None
                )
                plt.errorbar(
                    x=promedios_mod_block["True_Trial"],
                    y=promedios_mod_block[var],
                    yerr=promedios_mod_block[f"{var}_std"],
                    fmt='none',
                    ecolor='black',
                    capsize=5
                )

                plt.title(f"{var} - {modalidad} - {block} - Sujeto: {sujeto}")
                plt.xlabel("True_Trial")
                plt.ylabel(var)
                plt.legend(title="Categoría")
                plt.savefig(os.path.join(var_dir, f"{modalidad}_{block}.png"))
                plt.close()

                # Crear archivo .txt con resultados
                result_txt = os.path.join(var_dir, f"{modalidad}_{block}.txt")
                with open(result_txt, "w") as f:
                    f.write(f"Modalidad: {modalidad}\n")
                    f.write(f"Block: {block}\n")
                    f.write(f"Variable: {var}\n")
                    f.write("\nDatos:\n")
                    f.write(df_mod_block[["True_Trial", "Trial_Unique_ID", var]].to_string(index=False))

                # Identificar outliers basado en Z-scores
                df_mod_block[f"zscore_{var}"] = zscore(df_mod_block[var].dropna())
                df_outliers = df_mod_block[df_mod_block[f"zscore_{var}"].abs() > 2]

                with open(result_txt, "a") as f:
                    if not df_outliers.empty:
                        f.write("\n\nOutliers detectados:\n")
                        f.write(df_outliers["Trial_Unique_ID"].to_string(index=False))

                # Agregar detalles de outliers
                for _, row in df_outliers.iterrows():
                    detalles_outliers.append({
                        "Trial_Unique_ID": row["Trial_Unique_ID"],
                        "Sujeto": sujeto,
                        "Modalidad": modalidad,
                        "True_Block": block,
                        "Variable": var,
                        "Motivo": f"Z-score alto ({row[f'zscore_{var}']:.2f})"
                    })

                outliers.extend(df_outliers["Trial_Unique_ID"].tolist())

# Crear DataFrame de detalles de outliers
df_detalles_outliers = pd.DataFrame(detalles_outliers)
outliers_file = os.path.join(output_dir, "H1_outliers_identificados.csv")
df_detalles_outliers.to_csv(outliers_file, index=False)

print("Outliers identificados guardados en:", outliers_file)
