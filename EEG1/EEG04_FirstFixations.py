#%%-----------------------------------------------------
#
#   Fondecyt 11200469 - Mayo 2025
#   En este Script intentaré Definir las "primeras Fijaciones"
#
#-----------------------------------------------------
Song = 'Script para Primeras Fijaciones'
print('-------------------------------')
print(Song)
print('-------------------------------')


import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import sys
import os
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

print("Current working directory:", os.getcwd())
eeg1_dir = os.path.abspath('./EEG1')  # O pon ruta absoluta manual

print("Intentando agregar:", eeg1_dir)
if eeg1_dir not in sys.path:
    sys.path.append(eeg1_dir)

import EEG00_HMod

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Py_Sujetos_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/SUJETOS/")

lista_sujetos = [f for f in os.listdir(Py_Sujetos_Dir) if f.startswith('P') and os.path.isdir(os.path.join(Py_Sujetos_Dir, f))] # Todos los "P" efectivos




for Sujeto in sorted(lista_sujetos):
    Py_Specific_Dir = os.path.join(Py_Sujetos_Dir, Sujeto, 'EEG')
    for mod in ["NI","RV"]:
        print('Procesando archivos para fijaciones para Sujeto',Sujeto)
        fixfile = os.path.join(Py_Specific_Dir, f"fixation_forMATLAB_{mod}.csv")
        trialfile = os.path.join(Py_Specific_Dir, f"trials_forMATLAB_{mod}.csv")
        H_Mod.enriquecer_fijaciones_firstfix(fixfile, trialfile, modalidad=mod)


#%% Unificar base de datos de primeras fijaciones
# Cargar codificación de grupos
grupo_path = os.path.join(Py_Processing_Dir, "A_INFINITE_BASAL_DF.xlsx")
grupo_df = pd.read_excel(grupo_path)  # columnas: CODIGO, GRUPO

# Inicializar base acumuladora
df_todos = pd.DataFrame()

# Buscar todos los sujetos
for sujeto in sorted(os.listdir(Py_Sujetos_Dir)):
    if not sujeto.startswith("P"):
        continue

    path_summary = os.path.join(Py_Sujetos_Dir, sujeto, "EEG", "fixation_forMATLAB_NI_group_summary.csv")
    if not os.path.exists(path_summary):
        continue

    try:
        df = pd.read_csv(path_summary)
        df['Sujeto'] = sujeto
        df['Modalidad'] = 'NI'

        # Filtrar solo los trials NaviHT3
        df = df[df['Real_Trial'].str.contains("NaviHT3", na=False)]

        df_todos = pd.concat([df_todos, df], ignore_index=True)

    except Exception as e:
        print(f"⚠️ Error leyendo {path_summary}: {e}")

# Agregar columna de grupo
df_todos = df_todos.merge(grupo_df[['CODIGO', 'Grupo']], left_on='Sujeto', right_on='CODIGO', how='left')
df_todos.drop(columns=['CODIGO'], inplace=True)

# Guardar archivo final
output_file = os.path.join(Py_Processing_Dir, "EEG04_FirstFixations.csv")
df_todos.to_csv(output_file, index=False)
print(f"✅ Archivo guardado: {output_file}")




# --- CONFIGURACIÓN INICIAL ---
df = df_todos.copy()
df['zona_Y'] = np.where(df['y_inicio'] > 0.6, 'superior', 'inferior')

# --- MÉTRICAS POR SUJETO ---
conteo_zona = df.groupby(['Sujeto', 'Grupo', 'zona_Y']).size().unstack(fill_value=0).reset_index()
conteo_zona['proporcion_superior'] = (
    conteo_zona.get('superior', 0) /
    (conteo_zona.get('superior', 0) + conteo_zona.get('inferior', 0))
)

duracion_promedio = df.groupby(['Sujeto', 'Grupo'])['duracion_total'].mean().reset_index(name='duracion_promedio')

df = df.sort_values(by=['Sujeto', 'first_start'])
df['prev_zona'] = df.groupby('Sujeto')['zona_Y'].shift(1)
df['cambio_zona'] = (df['zona_Y'] != df['prev_zona']) & (~df['prev_zona'].isna())
saltos_por_sujeto = df.groupby(['Sujeto', 'Grupo'])['cambio_zona'].sum().reset_index(name='n_saltos')

resumen = conteo_zona.merge(duracion_promedio, on=['Sujeto', 'Grupo'])
resumen = resumen.merge(saltos_por_sujeto, on=['Sujeto', 'Grupo'])

# --- VISUALIZACIONES ---
sns.set(style='whitegrid')

# Dispersión espacial
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='x_inicio', y='y_inicio',
                size='duracion_total', hue='n_fijaciones',
                palette='viridis', edgecolor='k', alpha=0.7)
plt.title("Distribución espacial de primeras fijaciones (Modalidad NI)")
plt.xlabel("Posición X (normalizada)")
plt.ylabel("Posición Y (normalizada)")
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.gca().invert_yaxis()
plt.legend(title='n_fijaciones', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# KDE eje Y
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='y_inicio', hue='Grupo', fill=True, common_norm=False)
plt.title("Distribución de primeras fijaciones en el eje Y por grupo (NI)")
plt.xlabel("Posición Y (normalizada)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot duración promedio
plt.figure(figsize=(8, 5))
sns.boxplot(data=resumen, x='Grupo', y='duracion_promedio')
plt.title("Duración promedio de grupos de fijación")
plt.grid(True)
plt.tight_layout()
plt.show()

# Violinplot proporción superior
plt.figure(figsize=(8, 5))
sns.violinplot(data=resumen, x='Grupo', y='proporcion_superior', inner='box')
plt.title("Proporción de fijaciones en zona superior")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot número de saltos
plt.figure(figsize=(8, 5))
sns.boxplot(data=resumen, x='Grupo', y='n_saltos')
plt.title("Número de saltos entre zonas superior/inferior")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- ESTADÍSTICAS ---

print("\n=== ESTADÍSTICAS ===\n")

# ANOVA dinámico
def try_anova(df, var):
    if df['Grupo'].nunique() >= 2:
        model = ols(f'{var} ~ C(Grupo)', data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        print(f"\nANOVA para {var}:\n", table)
    else:
        print(f"❌ No se puede realizar ANOVA para {var}: menos de 2 grupos.")

try_anova(resumen, 'proporcion_superior')
try_anova(resumen, 'duracion_promedio')
try_anova(resumen, 'n_saltos')

# T-test entre MPPP y Control para proporción_superior
mppp = resumen[resumen['Grupo'] == 'MPPP']['proporcion_superior']
control = resumen[resumen['Grupo'] == 'Control']['proporcion_superior']

if len(mppp) >= 2 and len(control) >= 2:
    t_stat, p_val = ttest_ind(mppp, control, equal_var=False)
    print(f"\nT-test MPPP vs Control (proporción_superior):\nt = {t_stat:.3f}, p = {p_val:.4f}")
else:
    print("❌ No se puede realizar t-test: MPPP o Control tienen menos de 2 sujetos.")

# --- OUTPUT FINAL ---
print("\n=== RESUMEN POR SUJETO ===")
print(resumen.head())

# Si deseas guardar:
# resumen.to_csv("EEG05_FirstFixations_Resumen.csv", index=False)

#%%
# Seleccionamos solo la primera fijación de cada grupo de fijaciones
primeras = df.sort_values(by=['Sujeto', 'group_id', 'first_start'])
primeras = primeras.drop_duplicates(subset=['Sujeto', 'group_id'])

# Asegurar que estén dentro del rango de pantalla
primeras = primeras[(primeras['x_inicio'].between(0, 1)) & (primeras['y_inicio'].between(0, 1))].copy()

# Aplicamos jitter para evitar problemas con puntos duplicados
jitter_scale = 0.01  # Puedes ajustar el ruido si es necesario
primeras['x_jitter'] = primeras['x_inicio'] + np.random.normal(0, jitter_scale, size=len(primeras))
primeras['y_jitter'] = primeras['y_inicio'] + np.random.normal(0, jitter_scale, size=len(primeras))

# Recortamos cualquier valor que se pase del límite tras el jitter
primeras['x_jitter'] = primeras['x_jitter'].clip(0, 1)
primeras['y_jitter'] = primeras['y_jitter'].clip(0, 1)

# Plot KDE por grupo
grupos = primeras['Grupo'].dropna().unique()
for grupo in sorted(grupos):
    subset = primeras[primeras['Grupo'] == grupo]
    if len(subset) < 5:
        print(f"❗ Grupo {grupo} tiene muy pocos datos para KDE. Saltando.")
        continue

    plt.figure(figsize=(8, 6))
    sns.kdeplot(
        data=subset,
        x='x_jitter', y='y_jitter',
        fill=True,
        cmap="viridis",
        bw_adjust=0.5,
        levels=100,
        thresh=0.05
    )
    plt.title(f"KDE primeras fijaciones - Grupo {grupo}")
    plt.xlabel("x (normalizado)")
    plt.ylabel("y (normalizado)")
    plt.xlim(0, 1)
    plt.ylim(0.6, 1)
    #plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
