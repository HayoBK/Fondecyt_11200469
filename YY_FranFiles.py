#-------------------------------------------
#
#   Mayo 23, 2024.
#   Cosas de la Fran
#
#-------------------------------------------
import pandas as pd

import os

import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
import numpy as np
from pathlib import Path
import socket
import scipy.stats as stats
import scikit_posthocs as sp
from statsmodels.stats.anova import AnovaRM


# Cargar el archivo Excel


home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Fran_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/2024-Fran/"

ruta_archivo = Fran_Dir + 'Base.xlsx'  # Reemplaza con la ruta de tu archivo
df = pd.read_excel(ruta_archivo)
df = df.apply(pd.to_numeric, errors='coerce')


# Obtener los nombres de las columnas y sus índices
columnas = df.columns
listado_columnas = [(i, columna) for i, columna in enumerate(columnas)]

# Imprimir el listado de nombres de columnas y sus índices
for indice, nombre in listado_columnas:
    print(f"{indice}: {nombre}")

# Identificar tipos de columnas
ordinales_escaleras = df.select_dtypes(include=['number']).columns
categoriales = df.select_dtypes(include=['object']).columns

# Estadísticos descriptivos para variables ordinales/escalares
estadisticos_ordinales_escaleras = df[ordinales_escaleras].describe().T
estadisticos_ordinales_escaleras['range'] = estadisticos_ordinales_escaleras['max'] - estadisticos_ordinales_escaleras['min']
estadisticos_ordinales_escaleras = estadisticos_ordinales_escaleras[['mean', 'std', 'min', 'max']]

# Frecuencia de categorías para variables categóricas
frecuencias_categoriales = {}
for columna in categoriales:
    frecuencias = df[columna].value_counts().reset_index()
    frecuencias.columns = [columna, 'Frecuencia']
    frecuencias['Porcentaje'] = (frecuencias['Frecuencia'] / df[columna].count()) * 100
    frecuencias.to_excel(Fran_Dir + f'frecuencias_{columna}.xlsx', index=False)

# Convertir los resultados de frecuencias categóricas a un DataFrame
frecuencias_df = pd.DataFrame(frecuencias_categoriales)

# Exportar a archivos Excel
estadisticos_ordinales_escaleras.to_excel((Fran_Dir + 'estadisticos_ordinales_escaleras.xlsx'), index=True)
frecuencias_df.to_excel((Fran_Dir + 'frecuencias_categoriales.xlsx'), index=True)

# Imprimir confirmación
print("Estadísticos descriptivos y frecuencias categóricas exportados a archivos Excel.")

# Calcular matriz de correlaciones de Spearman
# Calcular matriz de correlaciones de Spearman
# Calcular matriz de correlaciones de Spearman

cols_of_interest = df.columns[43:59]
correlacion = pd.DataFrame(index=ordinales_escaleras, columns=ordinales_escaleras)
p_values = pd.DataFrame(index=ordinales_escaleras, columns=ordinales_escaleras)
effect_sizes = pd.DataFrame(index=ordinales_escaleras, columns=ordinales_escaleras)
significant_results = []

nan_cases = []

for col in ordinales_escaleras:
    for row in ordinales_escaleras:
        if col != row:
            # Eliminar filas con valores faltantes para el par de variables
            valid_data = df[[col, row]].dropna()
            if valid_data.shape[0] > 1:  # Necesitamos al menos dos filas para calcular la correlación
                corr_value, p_value = stats.spearmanr(valid_data[col], valid_data[row])
                correlacion.loc[row, col] = corr_value
                p_values.loc[row, col] = p_value
                effect_sizes.loc[row, col] = abs(corr_value)  # Tamaño del efecto como valor absoluto de la correlación
                if pd.isna(p_value):
                    nan_cases.append((row, col, "NaN p-value"))
                if p_value < 0.05:
                    significant_results.append((row, col, f"{corr_value:.2f}", f"{p_value:.3f}", f"{abs(corr_value):.2f}"))
            else:
                correlacion.loc[row, col] = None
                p_values.loc[row, col] = None
                effect_sizes.loc[row, col] = None
                nan_cases.append((row, col, "Insufficient data"))
        else:
            correlacion.loc[row, col] = None
            p_values.loc[row, col] = None
            effect_sizes.loc[row, col] = None

# Crear una tercera hoja con los valores combinados
combined = pd.DataFrame(index=ordinales_escaleras, columns=ordinales_escaleras)
for col in ordinales_escaleras:
    for row in ordinales_escaleras:
        if col != row:
            corr_value = correlacion.loc[row, col]
            p_value = p_values.loc[row, col]
            if corr_value is not None and p_value is not None:
                combined.loc[row, col] = f"{corr_value:.2f} ({p_value:.3f})"
            else:
                combined.loc[row, col] = None
        else:
            combined.loc[row, col] = None

# Crear una cuarta hoja con los casos NaN y errores
nan_cases_df = pd.DataFrame(nan_cases, columns=['Variable 1', 'Variable 2', 'Problema'])

# Crear una hoja adicional con correlaciones significativas
significant_results_df = pd.DataFrame(significant_results, columns=['Variable 1', 'Variable 2', 'Correlación', 'P-Value', 'Tamaño del Efecto'])

# Exportar la matriz de correlaciones, p-values, la hoja combinada, tamaños del efecto, casos NaN y resultados significativos a un archivo Excel
with pd.ExcelWriter(Fran_Dir + 'correlaciones_spearman_filtradas.xlsx') as writer:
    correlacion.to_excel(writer, sheet_name='Correlaciones')
    p_values.to_excel(writer, sheet_name='P-Values')
    combined.to_excel(writer, sheet_name='Correlacion y P-Value')
    effect_sizes.to_excel(writer, sheet_name='Tamaño del Efecto')
    nan_cases_df.to_excel(writer, sheet_name='Casos NaN y Errores', index=False)
    significant_results_df.to_excel(writer, sheet_name='Resultados Significativos', index=False)

# Imprimir confirmación
print("Matriz de correlaciones de Spearman y p-values exportados a un archivo Excel.")

# ANOVA Hayo

coparentalidad_momentos = df.iloc[:, 46:49]
coparentalidad_momentos.columns = ['Momento_1', 'Momento_2', 'Momento_3']

# Eliminar filas con valores faltantes
coparentalidad_momentos = coparentalidad_momentos.dropna()

# Preparar los datos para ANOVA de medidas repetidas
coparentalidad_melted = coparentalidad_momentos.melt(var_name='Momento', value_name='Coparentalidad', ignore_index=False)
coparentalidad_melted['Sujetos'] = coparentalidad_melted.index

# Realizar ANOVA de medidas repetidas
anova_model = AnovaRM(coparentalidad_melted, 'Coparentalidad', 'Sujetos', within=['Momento'])
anova_results = anova_model.fit()

# Mostrar los resultados del ANOVA
print('resultados ANOVA: chan chan')
print(anova_results)
ANOVA_Dir = Fran_Dir + 'ANOVA/'
# Generar el gráfico de boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Momento', y='Coparentalidad', data=coparentalidad_melted)
plt.title('Boxplot de Coparentalidad en Tres Momentos')
plt.xlabel('Momento')
plt.ylabel('Coparentalidad')
plt.savefig(ANOVA_Dir + 'boxplot_coparentalidad.png')

plt.show()
ANOVA_Dir = Fran_Dir + 'ANOVA/'
# Exportar los resultados del ANOVA a un archivo Excel
anova_results_summary = anova_results.summary().tables[0]
anova_results_df = pd.DataFrame(anova_results_summary)
anova_results_df.to_excel(ANOVA_Dir + 'anova_coparentalidad.xlsx', index=False)

# Guardar el gráfico como archivo PNG
# Analisis de caso

Fran_Dir = Fran_Dir + 'CASO/'


coparentalidad_momentos = df.iloc[:, 46:49]
coparentalidad_momentos.columns = ['Momento_1', 'Momento_2', 'Momento_3']
otras_variables = df.iloc[:, 49:59]

# Calcular las diferencias entre los tres momentos
df['Diferencia_M1_M2'] = coparentalidad_momentos['Momento_2'] - coparentalidad_momentos['Momento_1']
df['Diferencia_M2_M3'] = coparentalidad_momentos['Momento_3'] - coparentalidad_momentos['Momento_2']
df['Diferencia_M1_M3'] = coparentalidad_momentos['Momento_3'] - coparentalidad_momentos['Momento_1']

# Realizar análisis de correlación entre las diferencias y las variables adicionales
correlaciones_diferencias = {}
p_values_diferencias = {}

for diferencia in ['Diferencia_M1_M2', 'Diferencia_M2_M3', 'Diferencia_M1_M3']:
    correlaciones = []
    p_values = []
    for variable in otras_variables.columns:
        # Eliminamos las filas con valores faltantes antes de calcular la correlación
        valid_data = df[[diferencia, variable]].dropna()
        if valid_data.shape[0] > 1:  # Necesitamos al menos dos filas para calcular la correlación
            correlacion, p_value = stats.pearsonr(valid_data[diferencia], valid_data[variable])
            correlaciones.append(correlacion)
            p_values.append(p_value)
        else:
            correlaciones.append(None)
            p_values.append(None)
    correlaciones_diferencias[diferencia] = correlaciones
    p_values_diferencias[diferencia] = p_values

# Convertir los resultados a DataFrames
correlaciones_df = pd.DataFrame(correlaciones_diferencias, index=otras_variables.columns)
p_values_df = pd.DataFrame(p_values_diferencias, index=otras_variables.columns)

# Filtrar resultados significativos (p < 0.05)
significativos_df = pd.DataFrame()
for diferencia in ['Diferencia_M1_M2', 'Diferencia_M2_M3', 'Diferencia_M1_M3']:
    significativos = p_values_df[diferencia] < 0.05
    temp_df = pd.DataFrame({
        'Correlacion': correlaciones_df[diferencia][significativos],
        'P-Value': p_values_df[diferencia][significativos]
    })
    temp_df['Diferencia'] = diferencia
    significativos_df = pd.concat([significativos_df, temp_df])

# Exportar resultados a archivos Excel
correlaciones_df.to_excel(Fran_Dir + 'correlaciones_diferencias.xlsx')
p_values_df.to_excel(Fran_Dir + 'p_values_diferencias.xlsx')
significativos_df.to_excel(Fran_Dir + 'correlaciones_significativas.xlsx')

# Generar gráficos de dispersión para las correlaciones significativas
for index, row in significativos_df.iterrows():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[row['Diferencia']], y=df[index])
    plt.title(f'Correlación entre {row["Diferencia"]} y {index}\n Correlacion: {row["Correlacion"]:.2f}, P-Value: {row["P-Value"]:.3f}')
    plt.xlabel(row['Diferencia'])
    plt.ylabel(index)
    plt.savefig(Fran_Dir + f'correlacion_{row["Diferencia"]}_{index}.png')
    plt.show()

# Imprimir confirmación
print("Análisis de diferencias individuales y correlaciones exportados a archivos Excel.")

df = df.apply(pd.to_numeric, errors='coerce')

df_clean = df.dropna(subset=df.columns[49:59])

# Seleccionar las competencias de la madre (variables 49 a 53) y del padre (variables 54 a 58) después de limpiar los datos
competencias_madre = df_clean.iloc[:, 49:54]
competencias_padre = df_clean.iloc[:, 54:59]

# Calcular los valores promedio de cada competencia para proporcionar una descripción más completa
promedios_madre = competencias_madre.mean()
promedios_padre = competencias_padre.mean()

# Realizar el test de Wilcoxon para cada par de variables
wilcoxon_results = {}
for madre, padre in zip(competencias_madre.columns, competencias_padre.columns):
    wilcoxon_result = stats.wilcoxon(df_clean[madre], df_clean[padre], zero_method='wilcox')
    wilcoxon_results[(madre, padre)] = wilcoxon_result

# Mostrar los resultados del test de Wilcoxon
for par, result in wilcoxon_results.items():
    print(f"Comparación entre {par[0]} (Madre) y {par[1]} (Padre): p-value = {result.pvalue}")

# Generar el gráfico de boxplot para la comparación visual
plt.figure(figsize=(12, 12))
sns.boxplot(data=pd.concat([competencias_madre, competencias_padre], axis=1))
plt.xticks(rotation=45)
plt.title('Comparación de Competencias entre Madre y Padre')
plt.xlabel('Competencias')
plt.ylabel('Valor')
plt.xticks(ticks=range(10), labels=competencias_madre.columns.tolist() + competencias_padre.columns.tolist(), rotation=45)
plt.savefig(Fran_Dir + 'boxplot_comparacion_madre_padre.png')
plt.show()

# Imprimir los valores promedio de cada competencia
print("\nValores promedio de las competencias de la madre:")
print(promedios_madre)
print("\nValores promedio de las competencias del padre:")
print(promedios_padre)

ttest_results = {}
for madre, padre in zip(competencias_madre.columns, competencias_padre.columns):
    ttest_result = stats.ttest_rel(df_clean[madre], df_clean[padre])
    ttest_results[(madre, padre)] = ttest_result

# Mostrar los resultados del test t de Student
for par, result in ttest_results.items():
    print(f"Comparación entre {par[0]} (Madre) y {par[1]} (Padre): p-value = {result.pvalue}")

# Generar el gráfico de boxplot para la comparación visual


# Imprimir los valores promedio de cada competencia
print("\nValores promedio de las competencias de la madre:")
print(promedios_madre)
print("\nValores promedio de las competencias del padre:")
print(promedios_padre)

print(' Todo listo bajo el sol')