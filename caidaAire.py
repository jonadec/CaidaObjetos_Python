import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint

# Parámetros del problema
m = 160  # Masa del objeto en libras
g = 32.2  # Aceleración debido a la gravedad en ft/s^2
k = 0.25  # Constante de proporcionalidad para la resistencia del aire en lb/(ft/s)
v0 = 0.0  # Velocidad inicial en ft/s
x0 = 0.0  # Altura inicial en ft
posicion_objetivo = 500
t = np.linspace(0, 6, 100) 
condiciones_iniciales = [v0, x0]

# Función para calcular la velocidad con resistencia del aire proporcional a la velocidad
def v_linear(t, m, g, k, v0):
    return (m * g / k) + (v0 - m * g / k) * np.exp(-(k / m) * t)

def modelo_aire(v, t, m, g, k):
    dvdt = g - (k/m) * v
    return dvdt

# Función para calcular la posición integrando la velocidad
def calcular_posicion(t, v, x0):
    dt = t[1] - t[0]
    # Integrar la velocidad para obtener la posición
    posicion = np.cumsum(v) * dt
    posicion += x0
    return posicion

# Función para calcular la posición analíticamente integrando la velocidad
def s_analitica(t, m, g, k, v0, x0):
    term1 = (m * g / k) * t
    term2 = (m / k) * (v0 - m * g / k) * np.exp(-k * t / m)
    term3 = (m / k) * (v0 - m * g / k)
    return term1 + x0 + term3 * (1 - np.exp(-k * t / m))

# Calcular la velocidad
velocidad = v_linear(t, m, g, k, v0)
solucion = odeint(modelo_aire, condiciones_iniciales, t, args=(m, g, k))

# Calcular la posición numérica
posicion_numerica = calcular_posicion(t, velocidad, x0)

# Calcular la posición analítica
posicion_analitica = s_analitica(t, m, g, k, v0, x0)

# Encontrar la velocidad y el tiempo en la posición 500 ft
indice = np.argmax(posicion_numerica >= posicion_objetivo)
velocidad_en_posicion_objetivo = velocidad[indice]
tiempo_en_posicion_objetivo = t[indice]

# Graficar la velocidad del objeto en función del tiempo
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, velocidad, label='Velocidad (analítica)', color='green')
plt.plot(t, solucion[:, 0], label='Velocidad (odeint)', color='orange', linestyle='--')
plt.axvline(t[indice], color='r', linestyle='--', label=f'Tiempo en posición {posicion_objetivo} ft')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (ft/s)')
plt.title('Velocidad del objeto en función del tiempo')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, posicion_analitica, label='Posición (analítica)', color='green')
plt.plot(t, posicion_numerica, label='Posición (integrada)', color='orange')
plt.axhline(posicion_objetivo, color='r', linestyle='--', label=f'Posición {posicion_objetivo} ft')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (ft)')
plt.title('Posición del objeto en función del tiempo')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Velocidad en la posición {posicion_objetivo} ft: {velocidad_en_posicion_objetivo:.2f} ft/s")
print(f"Tiempo en la posición {posicion_objetivo} ft: {tiempo_en_posicion_objetivo:.2f} s")
