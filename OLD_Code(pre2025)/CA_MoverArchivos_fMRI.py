from pathlib import Path
import shutil

# Define la ruta base
base_path = Path.home() / "OneDrive" / "2-Casper" / "00-CurrentResearch" / "001-FONDECYT_11200469" / "002-LUCIEN" / "SUJETOS"

# Ruta de la carpeta de backup
backup_path = base_path / "BackupPres"

# Encuentra todas las carpetas que comienzan con 'P' seguido de números
subject_folders = [folder for folder in base_path.glob('P*') if folder.is_dir()]

# Itera sobre cada carpeta de sujeto
for folder in subject_folders:
    # Crea la subcarpeta 'fMRI_Presentation_Log' si no existe
    fmri_log_folder = folder / "fMRI_Presentation_Log"
    fmri_log_folder.mkdir(exist_ok=True)

    # Mueve los archivos correspondientes desde BackupPres
    subject_code = folder.name  # Ejemplo: 'P01'
    for file in backup_path.glob(subject_code + '*'):  # Busca archivos que comiencen con el código del sujeto
        shutil.move(str(file), str(fmri_log_folder))  # Mueve los archivos

print("Los archivos han sido movidos correctamente.")