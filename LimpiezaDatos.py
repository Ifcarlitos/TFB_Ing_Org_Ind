import pandas as pd
import numpy as np

#Url de los datos
df_ventas = pd.read_csv('./Datos_Principales/data_ventas.csv')
df_compras = pd.read_csv('./Datos_Principales/data_compras.csv')
df_movimientos_inventario = pd.read_csv('./Datos_Principales/data_movimientos_inventario.csv')
df_productos = pd.read_csv('./Datos_Principales/data_productos.csv')

#Limpieza de datos de Ventas

#Imprimir las columnas
# for col in df_ventas.columns:
#     print(col)

#Eliminar columnas innecesarias.
# Las necesarias son: ['Document No_', 'Line No_', 'Sell-to Customer No_', 'Type',
#                                    'No_', 'Location Code', 'Posting Date', 'Shipment Date', 'Description',
#                                    'Unit of Measure', 'Quantity', 'Unit Price', 'Unit Cost (LCY)', 'Line Discount _',
#                                       'Line Discount Amount', 'Amount', 'Amount Including VAT', 'Item Category Code']

df_ventas_limpio = df_ventas[['Document No_', 'Line No_', 'Sell-to Customer No_', 'Type',
                                   'No_', 'Location Code', 'Posting Date', 'Shipment Date', 'Description',
                                   'Unit of Measure', 'Quantity', 'Unit Price', 'Unit Cost (LCY)', 'Line Discount _',
                                      'Line Discount Amount', 'Amount', 'Amount Including VAT', 'Item Category Code']].copy()
# Csv de Ventas Limpio
df_ventas_limpio.to_csv('./Datos_Limpios/ventas_limpio.csv', index=False)

#Limpieza de datos de Compras

#Imprimir las columnas
# for col in df_compras.columns:
#     print(col)

#Eliminar columnas innecesarias.
# Las necesarias son: ['Document No_', 'Line No_', 'Buy-from Vendor No_', 'Type',
#                                    'No_', 'Location Code', 'Posting Date', 'Expected Receipt Date', 'Description',
#                                    'Unit of Measure', 'Quantity', 'Unit Cost', 'Line Discount _',
#                                       'Line Discount Amount', 'Amount', 'Item Category Code']

df_compras_limpio = df_compras[['Document No_', 'Line No_', 'Buy-from Vendor No_', 'Type',
                                      'No_', 'Location Code', 'Posting Date', 'Expected Receipt Date', 'Description',
                                      'Unit of Measure', 'Quantity', 'Unit Cost', 'Line Discount _',
                                          'Line Discount Amount', 'Amount', 'Item Category Code']].copy()

# # Csv de Compras Limpio
df_compras_limpio.to_csv('./Datos_Limpios/compras_limpio.csv', index=False)

#Limpieza de datos de Movimientos de Inventario

# Imprimir las columnas
# for col in df_movimientos_inventario.columns:
#     print(col)

#Eliminar columnas innecesarias.

# Las necesarias son: Entry No_
# Item No_
# Posting Date
# Entry Type
# Source No_
# Document No_
# Description
# Location Code
# Quantity

df_movimientos_inventario_limpio = df_movimientos_inventario[['Entry No_', 'Item No_', 'Posting Date', 'Entry Type', 'Source No_', 'Document No_', 'Description', 'Location Code', 'Quantity']].copy()

# Csv de Movimientos de Inventario Limpio
df_movimientos_inventario_limpio.to_csv('./Datos_Limpios/movimientos_inventario_limpio.csv', index=False)

#Limpieza de datos de Productos

#Imprimir las columnas
# for col in df_productos.columns:
#     print(col)

#Eliminar columnas innecesarias.
# Las necesarias son: ['No_', 'Description', 'Base Unit of Measure', 'Item Category Code']

df_productos_limpio = df_productos[['No_', 'Description', 'Base Unit of Measure', 'Item Category Code']].copy()

df_productos_limpio.loc[:, 'Cantidad en Inventario'] = np.nan

# Calcular la cantidad en inventario
for index, row in df_productos_limpio.iterrows():
    cantidad = df_movimientos_inventario_limpio[df_movimientos_inventario_limpio['Item No_'] == row['No_']]['Quantity'].sum()
    df_productos_limpio.loc[index, 'Cantidad en Inventario'] = cantidad

# Csv de Productos Limpio
df_productos_limpio.to_csv('./Datos_Limpios/productos_limpio.csv', index=False)

# Agregar la columna de typo venta a los datos de ventas
df_ventas_limpio.loc[:, 'Tipo'] = 'Venta'

# Agregar la columna de tipo compra a los datos de compras
df_compras_limpio.loc[:, 'Tipo'] = 'Compra'

# Unir los datos de ventas y compras
df_ventas_compras = pd.concat([df_ventas_limpio, df_compras_limpio])

# Csv de Ventas y Compras
df_ventas_compras.to_csv('./Datos_Limpios/ventas_compras.csv', index=False)

#Comprobación:
# Numero de registro de ventas
num_ventas = len(df_ventas_limpio)
print('Numero de registros de ventas:', num_ventas)
# Numero de registro de compras
num_compras = len(df_compras_limpio)
print('Numero de registros de compras:', num_compras)
# Numero de registros de ventas y compras
num_ventas_compras = len(df_ventas_compras)
print('Numero de registros de ventas y compras:', num_ventas_compras)

# Agregar columnas de productos a los datos de ventas y compras
df_ventas_compras_productos = pd.merge(df_ventas_compras, df_productos_limpio, left_on='No_', right_on='No_', how='left')

# Csv de Ventas y Compras con Productos
df_ventas_compras_productos.to_csv('./Datos_Limpios/ventas_compras_productos.csv', index=False)