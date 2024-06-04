import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parámetros del problema
m = 160  # Masa del objeto en libras
g = 32.2  # Aceleración debido a la gravedad en ft/s^2
k = 0.1  # Constante de proporcionalidad para la resistencia del líquido en lb/(ft/s)^2
v0 = 178.76  # Velocidad inicial en ft/s
x0 = 500  # Altura inicial en ft
posicion_objetivo = 575
t = np.linspace(0, 0.5, 100) 
condiciones_iniciales = [v0, x0]

def modelo_liquido(v, t, m, g, k):
    dvdt = g - (k/m) * v**2
    return dvdt


# Función para calcular la velocidad en el líquido con resistencia cuadrática
def velocity(t, m, g, k, V0):
    term1 = np.sqrt((m * g) / k)
    term2 = np.sqrt((k * g) / m) * t
    term3 = np.arctanh(np.sqrt((k / m) / g) * V0)
    return term1 * np.tanh(term2 + term3)

# Función para calcular la posición integrando la velocidad
def calcular_posicion(t, v, x0):
    dt = t[1] - t[0]
    posicion = np.cumsum(v) * dt
    posicion += x0
    return posicion

# Función para calcular la posición en función del tiempo de forma analítica
def s_analitica(t, m, g, k, V0, X0):
    term1 = np.sqrt((g * k) / m) * t
    term2 = np.sqrt(k / (m * g)) * V0
    term3 = np.cosh(term1 + np.arctanh(term2))
    term4 = np.cosh(np.arctanh(term2))
    term5 = np.log(term3 / term4)
    posicion = X0 + (m / k) * term5
    return posicion 

# Calcular la posición usando la función analítica
posicion_analitica = s_analitica(t, m, g, k, v0, x0)

# Calcular la velocidad
velocidad = velocity(t, m, g, k, v0)
solucion = odeint(modelo_liquido, condiciones_iniciales, t, args=(m, g, k))

# Calcular la posición
posicion = calcular_posicion(t, velocidad, x0)

# Encontrar la velocidad y el tiempo en la posición 575 ft
indice = np.argmax(posicion >= posicion_objetivo)
velocidad_en_posicion_objetivo = velocidad[indice]
tiempo_en_posicion_objetivo = t[indice]

# Graficar la velocidad y la posición del objeto en función del tiempo
plt.figure(figsize=(12, 6))

# Graficar la velocidad
plt.subplot(1, 2, 1)
plt.plot(t, velocidad, label='Velocidad (anlítica)', color='green')
plt.plot(t, solucion[:, 0], label='Velocidad (odeint)',color='orange', linestyle='--')
plt.axvline(t[indice], color='r', linestyle='--', label=f'Tiempo en posición {posicion_objetivo} ft')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (ft/s)')
plt.title('Velocidad del objeto en función del tiempo (líquido)')
plt.grid(True)
plt.legend()

# Graficar la posición
plt.subplot(1, 2, 2)
plt.plot(t, posicion_analitica, label='Posición (analítica)', color='green')
plt.plot(t, posicion, label='Posición (integrada)', color='orange')
plt.axhline(posicion_objetivo, color='r', linestyle='--', label=f'Posición {posicion_objetivo} ft')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (ft)')
plt.title('Posición del objeto en función del tiempo (líquido)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Velocidad en la posición {posicion_objetivo} ft: {velocidad_en_posicion_objetivo:.2f} ft/s")
print(f"Tiempo en la posición {posicion_objetivo} ft: {tiempo_en_posicion_objetivo:.2f} s")
