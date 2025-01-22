# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 21, Martes.
# Vamos de lleno a armar el Paper Numero 2.
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/")


file_path = Py_Processing_Dir + "A_INFINITE_BASAL_DF.xlsx"
codex_df = pd.read_excel(file_path)
file_path = Py_Processing_Dir + "F_Fixations_4All.csv"
fix_df = pd.read_csv(file_path)
file_path = Py_Processing_Dir + "H_SimianMaze_ShortDf_Normalized.csv"
navi_df = pd.read_csv(file_path)

#------------------------------------------------------------------------
# Exploratorio, veamos que pasa con las fijaciones antes de ponernos
# a escribir en serio.

fix_df = fix_df.sort_values(by=['Sujeto', 'fixation_id']).drop_duplicates(subset=['Sujeto', 'fixation_id'], keep='first')

# Añadir la columna Grupo a ffix_df basado en codex_df
fix_df = fix_df.merge(codex_df[['CODIGO', 'Grupo']], how='left', left_on='Sujeto', right_on='CODIGO')
fix_df.drop(columns=['CODIGO'], inplace=True)

# Filtrar por Modalidad "No Inmersivo" y True_Block en HiddenTarget_1, HiddenTarget_2, HiddenTarget_3
ffix_df = fix_df[(fix_df['Modalidad'] == 'No Inmersivo') &
                         (fix_df['True_Block'].isin(['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']))]
ffix_df = ffix_df[(ffix_df['on_surf'] == True)]
#------------------------------------------------------------------------
# Gráfico de boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=ffix_df, x='Grupo', y='norm_pos_y', hue='True_Block')
plt.title('Distribución de norm_por_y por Grupo y True_Block')
plt.xlabel('Grupo')
plt.ylabel('norm_pos_y')
plt.legend(title='True_Block')
plt.tight_layout()
plt.show()

true_blocks = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
grupos = ffix_df['Grupo'].unique()

# ------------------------------------------------------------------------
# Calcular el promedio de fijaciones por minuto por True_Block y Grupo
results = []
for block in ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']:
    for grupo in ffix_df['Grupo'].unique():
        data = ffix_df[(ffix_df['True_Block'] == block) & (ffix_df['Grupo'] == grupo)]

        if data.empty:
            continue

        # Calcular estadísticas para cada sujeto, bloque y trial
        grouped_data = data.groupby(['Sujeto', 'True_Trial'])
        trial_stats = grouped_data['world_timestamp'].apply(lambda x: (x.max() - x.min(), len(x)))
        trial_stats = trial_stats.apply(lambda x: (x[1] / (x[0] / 60)) if x[0] > 0 else None).dropna()

        # Promediar por sujeto y luego agregar al resultado general
        subject_mean = trial_stats.groupby(level=0).mean()
        results.extend([{'True_Block': block, 'Grupo': grupo, 'Fixations_per_Minute': val} for val in subject_mean])

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# ------------------------------------------------------------------------
# Gráfico de boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x='Grupo', y='Fixations_per_Minute', hue='True_Block')
plt.title('Promedio de Fijaciones por Minuto por Grupo y True_Block')
plt.xlabel('Grupo')
plt.ylabel('Fixaciones por Minuto')
plt.legend(title='True_Block')
plt.tight_layout()
plt.show()

#%% HardCore Calculations
for block in true_blocks:
    for grupo in grupos:
        plt.figure(figsize=(8, 6))
        data = ffix_df[(ffix_df['True_Block'] == block) & (ffix_df['Grupo'] == grupo)]

        if data.empty:
            print(f"No hay datos para True_Block={block} y Grupo={grupo}")
            continue

        sns.kdeplot(x=data['norm_pos_x'], y=data['norm_pos_y'], fill=True, cmap='viridis', cbar=True, bw_adjust=0.5)
        plt.title(f"Mapa de Calor: {block} - Grupo {grupo}")
        plt.xlabel("norm_pos_x")
        plt.ylabel("norm_pos_y")
        plt.tight_layout()
        plt.show()