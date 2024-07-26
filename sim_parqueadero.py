import simpy
import numpy as np
import matplotlib.pyplot as plt

# Definir parámetros
TIEMPOS_SERVICIO = {
    'rapido': 1,
    'normal': 3,
    'lento': 4,
    'muy_lento': 6
}
PROBABILIDADES_USUARIO = {
    'rapido': 0.25,
    'normal': 0.20,
    'lento': 0.35,
    'muy_lento': 0.20
}
TIEMPOS_LLEGADA = {
    'rapido': 3,
    'normal': 3,
    'lento': 5,
    'muy_lento': 7
}
NUM_REPLICAS = 50
UMBRAL_TIEMPO_ESPERA = 5  # Umbral de tiempo de espera aceptable (en minutos)
UMBRAL_ESTABILIDAD = 0.01  # Umbral de estabilidad para la desviación estándar acumulativa
TIEMPO_TRANSITORIO = 1000  # Tiempo transitorio a eliminar (en minutos)

def llegada_usuarios(env, cajero, tiempos_atencion_cajero, contador_usuarios, contador_usuarios_cajero):
    while True:
        # Determinar el tipo de usuario basado en las probabilidades
        tipo = np.random.choice(list(PROBABILIDADES_USUARIO.keys()), p=list(PROBABILIDADES_USUARIO.values()))
        tiempo_entre_llegadas = np.random.exponential(TIEMPOS_LLEGADA[tipo])
        yield env.timeout(tiempo_entre_llegadas)
        env.process(atender_usuario(env, cajero, tipo, tiempos_atencion_cajero, contador_usuarios, contador_usuarios_cajero))

def atender_usuario(env, cajero, tipo, tiempos_atencion, contador_usuarios, contador_usuarios_cajero):
    with cajero.request() as request:
        yield request
        tiempo_servicio = np.random.exponential(TIEMPOS_SERVICIO[tipo])
        yield env.timeout(tiempo_servicio)
        
        # Recolectar estadísticas por cajero
        tiempos_atencion.append((env.now, tiempo_servicio))
        
        # Incrementar contador de usuarios atendidos por tipo y por cajero
        contador_usuarios[tipo] += 1
        contador_usuarios_cajero[tipo] += 1

def ejecutar_simulacion():
    env = simpy.Environment()
    num_cajeros = 3
    cajeros = [simpy.Resource(env, capacity=1) for _ in range(num_cajeros)]

    tiempos_atencion_cajero = [[] for _ in range(num_cajeros)]
    contador_usuarios = {tipo: 0 for tipo in TIEMPOS_SERVICIO.keys()}
    contador_usuarios_cajero = [{tipo: 0 for tipo in TIEMPOS_SERVICIO.keys()} for _ in range(num_cajeros)]

    for i in range(num_cajeros):
        env.process(llegada_usuarios(env, cajeros[i], tiempos_atencion_cajero[i], contador_usuarios, contador_usuarios_cajero[i]))

    env.run(until=10000)  # Tiempo de simulación en minutos

    # Eliminar el estado transitorio
    tiempos_atencion_promedio = [
        np.mean([t[1] for t in tiempos if t[0] > TIEMPO_TRANSITORIO]) for tiempos in tiempos_atencion_cajero if tiempos
    ]
    return tiempos_atencion_promedio, contador_usuarios_cajero, tiempos_atencion_cajero

# Realizar múltiples réplicas y recolectar resultados
resultados_replicas = [ejecutar_simulacion() for _ in range(NUM_REPLICAS)]

# Separar resultados de tiempos y contadores de usuarios
tiempos_atencion_promedio_replicas = [result[0] for result in resultados_replicas]
contadores_usuarios_cajero_replicas = [result[1] for result in resultados_replicas]
tiempos_atencion_cajero_replicas = [result[2] for result in resultados_replicas]

# Calcular los promedios y desviaciones estándar por réplica
promedios_replicas_antes = [np.mean([t[1] for tiempos in replica for t in tiempos]) for replica in tiempos_atencion_cajero_replicas]
desviaciones_replicas_antes = [np.std([t[1] for tiempos in replica for t in tiempos]) for replica in tiempos_atencion_cajero_replicas]

promedios_replicas_despues = [np.mean(tiempos) for tiempos in tiempos_atencion_promedio_replicas]
desviaciones_replicas_despues = [np.std(tiempos) for tiempos in tiempos_atencion_promedio_replicas]

# Calcular desviación estándar acumulativa después de eliminar el estado transitorio
desviaciones_acumulativas_despues = [np.std(promedios_replicas_despues[:i+1]) for i in range(NUM_REPLICAS)]

# Calcular el total de usuarios por tipo para cada cajero en todas las réplicas
total_usuarios_cajero = [{tipo: sum(replica[i][tipo] for replica in contadores_usuarios_cajero_replicas) / NUM_REPLICAS
                          for tipo in TIEMPOS_SERVICIO.keys()} for i in range(3)]

# Graficar los promedios y desviaciones estándar antes de eliminar el estado transitorio
plt.figure(figsize=(12, 6))
plt.plot(promedios_replicas_antes, label='Promedio antes de eliminar el estado transitorio')
plt.fill_between(range(NUM_REPLICAS), 
                 [p - d for p, d in zip(promedios_replicas_antes, desviaciones_replicas_antes)], 
                 [p + d for p, d in zip(promedios_replicas_antes, desviaciones_replicas_antes)], 
                 color='b', alpha=0.2, label='Desviación estándar antes')
plt.xlabel('Número de réplicas')
plt.ylabel('Tiempo promedio de atención (minutos)')
plt.title('Convergencia del tiempo promedio de atención antes de eliminar el estado transitorio')
plt.legend()
plt.grid(True)
plt.show()

# Graficar los promedios y desviaciones estándar después de eliminar el estado transitorio
plt.figure(figsize=(12, 6))
plt.plot(promedios_replicas_despues, label='Promedio después de eliminar el estado transitorio')
plt.fill_between(range(NUM_REPLICAS), 
                 [p - d for p, d in zip(promedios_replicas_despues, desviaciones_replicas_despues)], 
                 [p + d for p, d in zip(promedios_replicas_despues, desviaciones_replicas_despues)], 
                 color='g', alpha=0.2, label='Desviación estándar después')
plt.xlabel('Número de réplicas')
plt.ylabel('Tiempo promedio de atención (minutos)')
plt.title('Convergencia del tiempo promedio de atención después de eliminar el estado transitorio')
plt.legend()
plt.grid(True)
plt.show()

# Graficar la desviación estándar acumulativa después de eliminar el estado transitorio
plt.figure(figsize=(12, 6))
plt.plot(desviaciones_acumulativas_despues, label='Desviación estándar acumulativa')
plt.xlabel('Número de réplicas')
plt.ylabel('Desviación estándar acumulativa')
plt.title('Desviación estándar acumulativa de tiempos de atención promedio')
plt.legend()
plt.grid(True)
plt.show()

# Graficar el promedio de usuarios atendidos por tipo para cada cajero
tipos_usuario = list(TIEMPOS_SERVICIO.keys())
promedios_usuarios_cajero = {tipo: [total_usuarios_cajero[i][tipo] for i in range(3)] for tipo in tipos_usuario}

fig, ax = plt.subplots(figsize=(12, 6))
width = 0.2  # Ancho de las barras
x = np.arange(3)  # Posición de los grupos de barras

for i, tipo in enumerate(tipos_usuario):
    ax.bar(x + i*width, promedios_usuarios_cajero[tipo], width, label=tipo)

ax.set_xlabel('Cajero')
ax.set_ylabel('Promedio de usuarios atendidos')
ax.set_title('Promedio de usuarios atendidos por tipo para cada cajero')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(['Cajero 1', 'Cajero 2', 'Cajero 3'])
ax.legend()
plt.grid(True)
plt.show()

# Determinar el número de réplicas necesarias para la estabilidad
for i in range(1, NUM_REPLICAS):
    if abs(desviaciones_acumulativas_despues[i] - desviaciones_acumulativas_despues[i-1]) < UMBRAL_ESTABILIDAD:  # Umbral de estabilidad
        replicas_estables = i + 1
        break
else:
    replicas_estables = NUM_REPLICAS

print(f'Número de réplicas necesarias para alcanzar la estabilidad: {replicas_estables}')

# Verificar si los tiempos de atención promedio están por debajo del umbral
tiempos_aceptables = all(promedio < UMBRAL_TIEMPO_ESPERA for promedio in promedios_replicas_despues[:replicas_estables])

# Evaluar la carga de trabajo y equilibrio entre cajeros
carga_equilibrada = all(abs(promedios_replicas_despues[i] - promedios_replicas_despues[j]) < 1 for i in range(len(promedios_replicas_despues)) for j in range(i+1, len(promedios_replicas_despues)))

# Decisión basada en los resultados
if tiempos_aceptables:
    decision = "La cantidad de tres cajeros es suficiente para suplir la demanda actual."
else:
    decision = "Se recomienda aumentar la cantidad de cajeros para mejorar los tiempos de atención."

print(decision)

# Imprimir resultados antes y después de eliminar el estado transitorio
print('\nResultados antes de eliminar el estado transitorio:')
for i, promedio in enumerate(promedios_replicas_antes):
    print(f'Replica {i+1}: Promedio tiempo de atención = {promedio:.2f} minutos, Desviación estándar = {desviaciones_replicas_antes[i]:.2f} minutos')

print('\nResultados después de eliminar el estado transitorio:')
for i, promedio in enumerate(promedios_replicas_despues):
    print(f'Replica {i+1}: Promedio tiempo de atención = {promedio:.2f} minutos, Desviación estándar = {desviaciones_replicas_despues[i]:.2f} minutos')

print(f'\nPromedio de usuarios de cada tipo por cajero en todas las réplicas:')
for i, contador in enumerate(total_usuarios_cajero):
    print(f'Cajero {i+1}: {contador}')

# Determinar el cajero con mayor y menor promedio de tiempo de atención
promedio_tiempo_atencion_cajero_global = np.mean(tiempos_atencion_promedio_replicas, axis=0)
cajero_max_promedio = np.argmax(promedio_tiempo_atencion_cajero_global) + 1
cajero_min_promedio = np.argmin(promedio_tiempo_atencion_cajero_global) + 1

print(f'Cajero con mayor promedio de tiempo de atención: Cajero {cajero_max_promedio}')
print(f'Cajero con menor promedio de tiempo de atención: Cajero {cajero_min_promedio}')

# Mostrar promedios de cada cajero en cada réplica
for replica_idx, tiempos_promedio in enumerate(tiempos_atencion_promedio_replicas):
    print(f'Replica {replica_idx + 1}: {["Cajero " + str(i + 1) + ": " + str(tiempos_promedio[i]) + " min" for i in range(len(tiempos_promedio))]}')

# Graficar los promedios de cada cajero en cada réplica
plt.figure(figsize=(12, 6))
for i in range(3):
    tiempos_promedio_cajero_i = [tiempos[i] for tiempos in tiempos_atencion_promedio_replicas]
    plt.plot(tiempos_promedio_cajero_i, label=f'Cajero {i+1}')

plt.xlabel('Número de réplicas')
plt.ylabel('Tiempo promedio de atención (minutos)')
plt.title('Promedio de tiempo de atención por cajero en cada réplica')
plt.legend()
plt.grid(True)
plt.show()
