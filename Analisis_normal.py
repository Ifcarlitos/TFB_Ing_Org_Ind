import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Url de los datos
df_ventas = pd.read_csv('./Datos_Principales/data_ventas.csv')
df_compras = pd.read_csv('./Datos_Principales/data_compras.csv')
df_movimientos_inventario = pd.read_csv('./Datos_Principales/data_movimientos_inventario.csv')
df_productos = pd.read_csv('./Datos_Principales/data_productos.csv')

# Numero de registros de cada tabla
print('Numero de registros de Ventas:', len(df_ventas))
print('Numero de registros de Compras:', len(df_compras))
print('Numero de registros de Movimientos de Inventario:', len(df_movimientos_inventario))
print('Numero de registros de Productos:', len(df_productos))

# Grafico productos mas vendidos
productos_mas_vendidos = df_ventas['No_'].value_counts().head(10)
productos_mas_vendidos.plot(kind='bar', title='Productos mas vendidos')

plt.show()

# Grafico productos mas comprados
productos_mas_comprados = df_compras['No_'].value_counts().head(10)
productos_mas_comprados.plot(kind='bar', title='Productos mas comprados')

plt.show()

# Grafico productos con mas movimientos de inventario
productos_mas_movimientos = df_movimientos_inventario['Item No_'].value_counts().head(10)
productos_mas_movimientos.plot(kind='bar', title='Productos con mas movimientos de inventario')

plt.show()

# Grafico de ventas por mes y año
df_ventas['mes'] = pd.to_datetime(df_ventas['Posting Date']).dt.month
df_ventas['año'] = pd.to_datetime(df_ventas['Posting Date']).dt.year
ventas_por_mes = df_ventas.groupby(['año', 'mes'])['Amount'].sum()
ventas_por_mes.plot(kind='bar', title='Ventas por mes y año')

plt.show()

# Grafico de compras por mes y año
df_compras['mes'] = pd.to_datetime(df_compras['Posting Date']).dt.month
df_compras['año'] = pd.to_datetime(df_compras['Posting Date']).dt.year
compras_por_mes = df_compras.groupby(['año', 'mes'])['Amount'].sum()
compras_por_mes.plot(kind='bar', title='Compras por mes y año')

plt.show()

# Grafico de movimientos de inventario por mes y año
df_movimientos_inventario['mes'] = pd.to_datetime(df_movimientos_inventario['Posting Date']).dt.month
df_movimientos_inventario['año'] = pd.to_datetime(df_movimientos_inventario['Posting Date']).dt.year
movimientos_por_mes = df_movimientos_inventario.groupby(['año', 'mes'])['Quantity'].sum()
movimientos_por_mes.plot(kind='bar', title='Movimientos de inventario por mes y año')

plt.show()

# Grafico de ventas por categoria
ventas_por_categoria = df_ventas['Item Category Code'].value_counts()
ventas_por_categoria.plot(kind='bar', title='Ventas por categoria')

plt.show()

# Grafico de compras por categoria
compras_por_categoria = df_compras['Item Category Code'].value_counts()
compras_por_categoria.plot(kind='bar', title='Compras por categoria')

plt.show()