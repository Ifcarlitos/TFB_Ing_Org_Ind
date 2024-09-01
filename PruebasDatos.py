import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

#Recoger datos de un archivo csv Datos_Finales_Analisis

ventas_df = pd.read_csv('./Datos_Finales_Analisis/ventas_df.csv')
compras_df = pd.read_csv('./Datos_Finales_Analisis/compras_df.csv')
productos_df = pd.read_csv('./Datos_Limpios/productos_limpio.csv')

#Recoger los modelos de regresión lineal del pkl
import pickle

with open('./Modelos/modelo_venta.pkl', 'rb') as file:
    ventas_model = pickle.load(file)

with open('./Modelos/modelo_compra.pkl', 'rb') as file:
    compras_model = pickle.load(file)

#Recoger Modelos svr

with open('./Modelos/modelo_venta_svr.pkl', 'rb') as file:
    ventas_svr = pickle.load(file)

with open('./Modelos/modelo_compra_svr.pkl', 'rb') as file:
    compras_svr = pickle.load(file)

#Recoger Modelo Cascada de ventas
with open('./Modelos/meta_classifier_cascade_Ventas.pkl', 'rb') as file:
    cascada_ventas = pickle.load(file)

#Recoger Modelo Cascada de compras
with open('./Modelos/meta_classifier_cascade_Compras.pkl', 'rb') as file:
    cascada_compras = pickle.load(file)

# #Revisar columnas necesarias para el modelo
# print('-----------------------------------Ventas-----------------------------------')
# for column in ventas_df.columns:
#     print(column)

# #Revisar columnas necesarias para el modelo
# print('-----------------------------------Compras-----------------------------------')
# for column in compras_df.columns:
#     print(column)

# Pruebas de los modelos
import pandas as pd
import numpy as np

# Cargar DataFrame de productos
productos_df = pd.read_csv('./Datos_Limpios/productos_limpio.csv')

# Extraer valores únicos de 'No_'
no_productos = productos_df['No_'].unique()

# Generar datos de prueba aleatorios
n_samples = 10  # Número de muestras
np.random.seed(42)  # Semilla para reproducibilidad

# Suponiendo que tenemos la lista de columnas usadas para entrenar el modelo (ajústalo según tus necesidades)
columnas_modelo = ['Amount', 'Year', 'Cantidad en Inventario', 'Day', 'Unit Price', 'Month'] + [f'No_{no}' for no in productos_df['No_'].unique()]

# Preparar datos de prueba
datos_prueba = pd.DataFrame({
    'Amount': np.random.uniform(100, 5000, size=n_samples),
    'Year': np.random.choice([2019, 2020, 2021], size=n_samples),
    'Cantidad en Inventario': np.random.randint(0, 100, size=n_samples),
    'Day': np.random.randint(1, 31, size=n_samples),
    'Unit Price': np.random.uniform(10, 1000, size=n_samples),
    'Month': np.random.choice(range(1, 13), size=n_samples),
    'No_': np.random.choice(no_productos, size=n_samples)
})

# Convertir 'No_' a variables dummy
datos_prueba = pd.get_dummies(datos_prueba, columns=['No_'])

# Asegurar que todas las columnas necesarias estén presentes
for col in columnas_modelo:
    if col not in datos_prueba.columns:
        datos_prueba[col] = 0

# Ordenar las columnas según el modelo
datos_prueba = datos_prueba[columnas_modelo]

# Hacer predicciones
y_pred_venta = ventas_model.predict(datos_prueba)

# Imprimir predicciones
print("Predicciones de Ventas:", y_pred_venta)


