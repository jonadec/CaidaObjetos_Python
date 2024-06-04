import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
g = 9.81  
v0 = 0.0  
h0 = 575  
t_max = np.sqrt(2 * h0 / g)  
t = np.linspace(0, t_max, 100)  

# Función para calcular la velocidad en caída libre
def velocidad_caida_libre(t, g, v0):
    return v0 + g * t

# Función para calcular la posición en caída libre
def posicion_caida_libre(t, g, h0, v0):
    return h0 + v0 * t - 0.5 * g * t**2

# Calcular la velocidad
velocidad = velocidad_caida_libre(t, g, v0)

# Calcular la posición
posicion = posicion_caida_libre(t, g, h0, v0)

# Graficar la velocidad del objeto en función del tiempo
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, velocidad, label='Velocidad (caída libre)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.title('Velocidad del objeto en función del tiempo (caída libre)')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, posicion, label='Posición (caída libre)', color='orange')
plt.axhline(0, color='r', linestyle='--', label='Posición del suelo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.title('Posición del objeto en función del tiempo (caída libre)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


tiempo_impacto = t_max
velocidad_impacto = velocidad_caida_libre(tiempo_impacto, g, v0)

print(f"Tiempo de impacto: {tiempo_impacto:.2f} s")
print(f"Velocidad de impacto: {velocidad_impacto:.2f} m/s")
