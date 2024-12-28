import pyautogui
import time
import socket

# Obtiene el nombre del host de la máquina actual
nombre_host = socket.gethostname()
print(nombre_host)

# Define código específico para cada host
if nombre_host == 'DESKTOP-PQ9KP6K':
    print("Ejecutando en el UChile")

a=0
delay = 900
def convertir_segundos(segundos_totales):
    dias = segundos_totales // (24 * 3600)  # Calcula el número de días
    segundos_totales = segundos_totales % (24 * 3600)  # Obtiene el resto de los segundos después de contar los días

    horas = segundos_totales // 3600  # Calcula el número de horas
    segundos_totales = segundos_totales % 3600  # Obtiene el resto de los segundos después de contar las horas

    minutos = segundos_totales // 60  # Calcula el número de minutos
    segundos = segundos_totales % 60  # Obtiene el resto de los segundos después de contar los minutos

    return dias, horas, minutos, segundos

# Ejemplo de uso
segundos = 100000  # Por ejemplo, 100,000 segundos
resultado = convertir_segundos(segundos)
print(f"{resultado[0]} días, {resultado[1]} horas, {resultado[2]} minutos, {resultado[3]} segundos")
while True:
    x=100
    a+=1
    if a % 2 == 0:  # Comprobamos si a es par
        x = 100
    else:  # Si a es impar
        x = -100
    pyautogui.moveRel(x, 0)  # Mueve el mouse ligeramente
    print('Ciclo: ',a)
    resultado = convertir_segundos((a*delay))
    print(f"{resultado[0]} días, {resultado[1]} horas, {resultado[2]} minutos, {resultado[3]} segundos")
    time.sleep(delay)  # Espera 10 minutos antes de moverlo nuevamente
