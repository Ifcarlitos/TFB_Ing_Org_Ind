import pyodbc
import pandas as pd
import numpy as np

# Configuración de la conexión
server = 'bcserver' 
database = 'tenant' 
username = 'admin' 
password = 'P@ssw0rd' 

# Cadena de conexión
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# Conectar
conn = pyodbc.connect(conn_str)

#Extraer Productos:

# Crear un cursor
cursor_productos = conn.cursor()

# Escribir una consulta SQL
query_productos = "SELECT * FROM [dbo].[CRONUS ES$Item$437dbf0e-84ff-417a-965d-ed2bb9650972]"

# Ejecutar la consulta
cursor_productos.execute(query_productos)

# Recuperar los resultados
rows_productos = cursor_productos.fetchall()

# Crear un Array de Productos
productos = []
for row in rows_productos:
    productos.append(row)

#Revisar las columnas
columns = [column[0] for column in cursor_productos.description]

# Crear un DataFrame de Productos
df_productos = pd.DataFrame(np.array(productos), columns=columns)

#Extraer Ventas:

# Crear un cursor
cursor_ventas = conn.cursor()

# Escribir una consulta SQL, incluyendo el filtro de Type = 2, que es el tipo linea de venta y 2 es la opción de producto.
query_ventas = "SELECT * FROM [dbo].[CRONUS ES$Sales Invoice Line$437dbf0e-84ff-417a-965d-ed2bb9650972] where Type = 2"

# Ejecutar la consulta
cursor_ventas.execute(query_ventas)

# Recuperar los resultados
rows_ventas = cursor_ventas.fetchall()

# Crear Array de Ventas
ventas = []
for row in rows_ventas:
    ventas.append(row)

#Revisar las columnas
columns = [column[0] for column in cursor_ventas.description]

# Crear un DataFrame de Ventas
df_ventas = pd.DataFrame(np.array(ventas), columns=columns)

# Extraer las Compras:

# Crear un cursor
cursor_compras = conn.cursor()

# Escribir una consulta SQL, incluyendo el filtro de Type = 2, que es el tipo linea de compra y 1 es la opción de producto.
query_compras = "SELECT * FROM [dbo].[CRONUS ES$Purch_ Inv_ Line$437dbf0e-84ff-417a-965d-ed2bb9650972] where Type = 2"

# Ejecutar la consulta
cursor_compras.execute(query_compras)

# Recuperar los resultados
rows_compras = cursor_compras.fetchall()

# Crear Array de Compras
compras = []
for row in rows_compras:
    compras.append(row)

#Revisar las columnas
columns = [column[0] for column in cursor_compras.description]

# Crear un DataFrame de Compras
df_compras = pd.DataFrame(np.array(compras), columns=columns)

# Extraer movimientos de inventario:

# Crear un cursor
cursor_movimientos_inventario = conn.cursor()

# Escribir una consulta SQL
query_movimientos_inventario = "SELECT * FROM [dbo].[CRONUS ES$Item Ledger Entry$437dbf0e-84ff-417a-965d-ed2bb9650972]"

# Ejecutar la consulta
cursor_movimientos_inventario.execute(query_movimientos_inventario)

# Recuperar los resultados
rows_movimientos_inventario = cursor_movimientos_inventario.fetchall()

# Crear Array de Movimientos de Inventario
movimientos_inventario = []
for row in rows_movimientos_inventario:
    movimientos_inventario.append(row)

#Revisar las columnas
columns = [column[0] for column in cursor_movimientos_inventario.description]

# Crear un DataFrame de Movimientos de Inventario
df_movimientos_inventario = pd.DataFrame(np.array(movimientos_inventario), columns=columns)

# Crear un archivo CSV con los datos extraidos
df_productos.to_csv('.\Datos_Principales\data_productos.csv', index=False)
df_ventas.to_csv('.\Datos_Principales\data_ventas.csv', index=False)
df_compras.to_csv('.\Datos_Principales\data_compras.csv', index=False)
df_movimientos_inventario.to_csv('.\Datos_Principales\data_movimientos_inventario.csv', index=False)

# Cerrar la conexión
conn.close()