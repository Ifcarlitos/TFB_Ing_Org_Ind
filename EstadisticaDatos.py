import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Url de los datos
df_ventas_compras = pd.read_csv('./Datos_Limpios/ventas_compras_productos.csv')
df_movimientos_inventario = pd.read_csv('./Datos_Limpios/movimientos_inventario_limpio.csv')
df_productos = pd.read_csv('./Datos_Limpios/productos_limpio.csv')

# Preprocesamiento de Datos:

# Tratar valores faltantes
ventas_clean = df_ventas_compras.fillna(method='ffill').fillna(method='bfill')

# Separar las transacciones de ventas y compras
ventas_df = ventas_clean[ventas_clean['Tipo'] == 'Venta'].copy()
compras_df = ventas_clean[ventas_clean['Tipo'] == 'Compra'].copy()

# Convertir las columnas de fechas
ventas_df['Posting Date'] = pd.to_datetime(ventas_df['Posting Date'])
compras_df['Posting Date'] = pd.to_datetime(compras_df['Posting Date'])

# Revisar los resultados de las primeras filas y la información de los DataFrames
# (ventas_df.head(), compras_df.head(), ventas_df.info(), compras_df.info())

# Añadir columnas de 'Month' y 'Year'
ventas_df['Month'] = ventas_df['Posting Date'].dt.month
ventas_df['Year'] = ventas_df['Posting Date'].dt.year
ventas_df['Day'] = ventas_df['Posting Date'].dt.day
compras_df['Month'] = compras_df['Posting Date'].dt.month
compras_df['Year'] = compras_df['Posting Date'].dt.year
compras_df['Day'] = compras_df['Posting Date'].dt.day

# Agrupar por mes y año y productos
ventas_por_mes_año = ventas_df.groupby(['Year', 'Month', 'No_']).size()
compras_por_mes_año = compras_df.groupby(['Year', 'Month', 'No_']).size()

# Crear gráficos para visualizar las tendencias mensuales de ventas y compras por producto
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

ventas_por_mes_año.unstack().plot(ax=ax[0], title='Ventas Mensuales por Producto', colormap='tab20')
compras_por_mes_año.unstack().plot(ax=ax[1], title='Compras Mensuales por Producto', colormap='tab20')

plt.tight_layout()

#Análisis de Correlación
# Seleccionar variables numéricas relevantes
correlaciones = ventas_df.corr()

# Crear un mapa de calor para visualizar las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor de Correlación entre Variables de Ventas')

# Correlacion compras
correlaciones_compras = compras_df.corr()

# Crear un mapa de calor para visualizar las correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlaciones_compras, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Mapa de Calor de Correlación entre Variables de Compras')

# Identificar las variables más correlacionadas con la cantidad vendida
correlacion_cantidad = correlaciones['Quantity'].sort_values(ascending=False)
print('Correlación de Cantidad Vendida:')
print(correlacion_cantidad)

# Borrar las columnas sin correlación
# Type                           NaN
# Line Discount _                NaN
# Line Discount Amount           NaN
# Buy-from Vendor No_            NaN
# Unit Cost                      NaN

ventas_df = ventas_df.drop(['Type', 'Line Discount _', 'Line Discount Amount', 
                            'Buy-from Vendor No_', 'Unit Cost'], axis=1)

# Identificar las variables más correlacionadas con la cantidad comprada
correlacion_cantidad_compras = correlaciones_compras['Quantity'].sort_values(ascending=False)
print('Correlación de Cantidad Comprada:')
print(correlacion_cantidad_compras)

# Borrar las columnas sin correlación
# Sell-to Customer No_           NaN
# Type                           NaN
# Unit Price                     NaN
# Unit Cost (LCY)                NaN
# Line Discount _                NaN
# Line Discount Amount           NaN
# Amount Including VAT           NaN

compras_df = compras_df.drop(['Sell-to Customer No_', 'Type', 'Unit Price',
                                'Unit Cost (LCY)', 'Line Discount _', 'Line Discount Amount',
                                'Amount Including VAT'], axis=1)

# Revisar Variables para el modelo:
y_venta = ventas_df['Quantity']
y_compra = compras_df['Quantity']

# Seleccionar las variables para el modelo de ventas

X_venta = ventas_df[['Amount', 'Year', 'Cantidad en Inventario', 
                    'Day', 'Unit Price', 'No_', 'Month']].copy()

#Convertir las columnas categóricas en variables dummy
X_venta = pd.get_dummies(X_venta, columns=['No_'])

#Seleccionar las variables para el modelo de compras

X_compra = compras_df[['Amount', 'Cantidad en Inventario', 'No_',
                          'Unit Cost', 'Month', 'Day']].copy()

#Convertir las columnas categóricas en variables dummy
X_compra = pd.get_dummies(X_compra, columns=['No_'])

#guardar los datos limpios en Datos_Finales_Analisis
ventas_df.to_csv('./Datos_Finales_Analisis/ventas_df.csv', index=False)
compras_df.to_csv('./Datos_Finales_Analisis/compras_df.csv', index=False)

#Dividir los datos en conjuntos de entrenamiento y prueba, para ventas y compras
from sklearn.model_selection import train_test_split

X_train_venta, X_test_venta, y_train_venta, y_test_venta = train_test_split(X_venta, y_venta, test_size=0.2, random_state=42)
X_train_compra, X_test_compra, y_train_compra, y_test_compra = train_test_split(X_compra, y_compra, test_size=0.2, random_state=42)

#-------------------------------------------------------------------------------------------------------------------------------

# Crear un modelo de regresión lineal para predecir la cantidad vendida
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

modelo_venta = LinearRegression()
modelo_venta.fit(X_train_venta, y_train_venta)

y_pred_venta = modelo_venta.predict(X_test_venta)

# Calcular el error cuadrático medio
mse_venta = mean_squared_error(y_test_venta, y_pred_venta)
print('Error Cuadrático Medio para Ventas:', mse_venta)
print('R2 Score para Ventas:', r2_score(y_test_venta, y_pred_venta))

# Crear un modelo de regresión lineal para predecir la cantidad comprada
modelo_compra = LinearRegression()
modelo_compra.fit(X_train_compra, y_train_compra)

y_pred_compra = modelo_compra.predict(X_test_compra)

# Calcular el error cuadrático medio
mse_compra = mean_squared_error(y_test_compra, y_pred_compra)
print('Error Cuadrático Medio para Compras:', mse_compra)
print('R2 Score para Compras:', r2_score(y_test_compra, y_pred_compra))

#-------------------------------------------------------------------------------------------------------------------------------

# Crear un modelo de regresión SVR para predecir la cantidad vendida y comprada
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Crear un pipeline que incluye estandarización de los datos y el modelo SVR
modelo_venta_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
modelo_compra_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

# Entrenar el modelo SVR para ventas
modelo_venta_svr.fit(X_train_venta, y_train_venta)
y_pred_venta_svr = modelo_venta_svr.predict(X_test_venta)

# Calcular el error cuadrático medio y R² para el modelo de ventas
mse_venta_svr = mean_squared_error(y_test_venta, y_pred_venta_svr)
r2_venta_svr = r2_score(y_test_venta, y_pred_venta_svr)

# Entrenar el modelo SVR para compras
modelo_compra_svr.fit(X_train_compra, y_train_compra)
y_pred_compra_svr = modelo_compra_svr.predict(X_test_compra)

# Calcular el error cuadrático medio y R² para el modelo de compras
mse_compra_svr = mean_squared_error(y_test_compra, y_pred_compra_svr)
r2_compra_svr = r2_score(y_test_compra, y_pred_compra_svr)

# Imprimir los resultados para ambos modelos
print('SVR para Ventas - MSE:', mse_venta_svr, 'R²:', r2_venta_svr)
print('SVR para Compras - MSE:', mse_compra_svr, 'R²:', r2_compra_svr)

#-------------------------------------------------------------------------------------------------------------------------------

# Combinació en Cascada de Modelos
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

myseed = 42 # 42 es el valor de la semilla aleatoria, que se utiliza para reproducir los resultados

#Ventas:

# Fem la definició del model de arbre de decisió amb una profunditat de 3
arbre_decisio_cascade = DecisionTreeRegressor(max_depth=3, random_state=myseed)
arbre_decisio_cascade.fit(X_train_venta, y_train_venta)
prediccio_arbre_cascade = arbre_decisio_cascade.predict(X_test_venta)

# Fem la definició del model K-Neighbors Classifier amb 2 veïns
knn_cascade = KNeighborsRegressor(n_neighbors=2)
knn_cascade.fit(X_train_venta, y_train_venta)
prediccio_knn_cascade = knn_cascade.predict(X_test_venta)

# Fem la definició del model SVM amb un gamma de 0.07
svm_cascade = SVR(gamma=0.07)
svm_cascade.fit(X_train_venta, y_train_venta)
prediccio_svm_cascade = svm_cascade.predict(X_test_venta)

# Agrupem les prediccions dels classificadors base en una matriu
X_meta_cascade = np.column_stack((prediccio_arbre_cascade, prediccio_knn_cascade, prediccio_svm_cascade))

# Fem la combinació de les prediccions dels classificadors base amb les dades de prova
X_cascade = np.column_stack((X_test_venta, X_meta_cascade))

# Ara definim el metaclassificador Gradient Boosting amb 20 arbres i una profunditat de 3
meta_classifier_cascade = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=myseed)

# Fem la validació creuada del metaclassificador amb les prediccions dels classificadors base i les dades originals:
cv_scores_cascade = cross_val_score(meta_classifier_cascade, X_cascade, y_test_venta, cv=5, scoring='accuracy')

# Fem el calcul de la mitjana i la desviació estàndard de la precisió:
mean_cv_score_cascade = cv_scores_cascade.mean()
std_cv_score_cascade = cv_scores_cascade.std()

print(f'Cascading Classifier - Validació Creuada (5-fold) - Precisió Mitjana: {mean_cv_score_cascade:.4f}, Desviació Estàndard: {std_cv_score_cascade:.4f}')

# Entrenem el metaclassificador amb les prediccions dels classificadors base i les dades originals:
meta_classifier_cascade.fit(X_cascade, y_test_venta)

# Fem la predicció del metaclassificador amb les dades de prova:
pred_meta_cascade = meta_classifier_cascade.predict(X_cascade)

# I fem el calcul de la precisió del metaclassificador:
test_accuracy_meta_cascade = meta_classifier_cascade.score(X_cascade, y_test_venta)

print(f'Cascading Classifier - Precisió sobre el Conjunt de Prova: {test_accuracy_meta_cascade:.4f}')

#Calcular r^2 para el modelo de ventas de cascada
r2_cascade = r2_score(y_test_venta, pred_meta_cascade)
print('R² para el modelo de ventas de cascada:', r2_cascade)
#error cuadrático medio para el modelo de ventas de cascada
mse_cascade = mean_squared_error(y_test_venta, pred_meta_cascade)
print('Error Cuadrático Medio para Ventas de Cascada:', mse_cascade)


#Compras:

# Fem la definició del model de arbre de decisió amb una profunditat de 3
arbre_decisio_cascade_compras = DecisionTreeRegressor(max_depth=3, random_state=myseed)
arbre_decisio_cascade_compras.fit(X_train_compra, y_train_compra)
prediccio_arbre_cascade_compras = arbre_decisio_cascade_compras.predict(X_test_compra)

# Fem la definició del model K-Neighbors Classifier amb 2 veïns
knn_cascade_compras = KNeighborsRegressor(n_neighbors=2)
knn_cascade_compras.fit(X_train_compra, y_train_compra)
prediccio_knn_cascade_compras = knn_cascade_compras.predict(X_test_compra)

# Fem la definició del model SVM amb un gamma de 0.07
svm_cascade_compras = SVR(gamma=0.07)
svm_cascade_compras.fit(X_train_compra, y_train_compra)
prediccio_svm_cascade_compras = svm_cascade_compras.predict(X_test_compra)

# Agrupem les prediccions dels classificadors base en una matriu
X_meta_cascade_compras = np.column_stack((prediccio_arbre_cascade_compras, prediccio_knn_cascade_compras, prediccio_svm_cascade_compras))

# Fem la combinació de les prediccions dels classificadors base amb les dades de prova
X_cascade_compras = np.column_stack((X_test_compra, X_meta_cascade_compras))

# Ara definim el metaclassificador Gradient Boosting amb 20 arbres i una profunditat de 3
meta_classifier_cascade_compras = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=myseed)

# Fem la validació creuada del metaclassificador amb les prediccions dels classificadors base i les dades originals:
cv_scores_cascade_compras = cross_val_score(meta_classifier_cascade_compras, X_cascade_compras, y_test_compra, cv=5, scoring='accuracy')

# Fem el calcul de la mitjana i la desviació estàndard de la precisió:
mean_cv_score_cascade_compras = cv_scores_cascade_compras.mean()
std_cv_score_cascade_compras = cv_scores_cascade_compras.std()

print(f'Cascading Classifier - Validació Creuada (5-fold) - Precisió Mitjana: {mean_cv_score_cascade_compras:.4f}, Desviació Estàndard: {std_cv_score_cascade_compras:.4f}')

# Entrenem el metaclassificador amb les prediccions dels classificadors base i les dades originals:
meta_classifier_cascade_compras.fit(X_cascade_compras, y_test_compra)

# Fem la predicció del metaclassificador amb les dades de prova:
pred_meta_cascade_compras = meta_classifier_cascade_compras.predict(X_cascade_compras)

# I fem el calcul de la precisió del metaclassificador:
test_accuracy_meta_cascade_compras = meta_classifier_cascade_compras.score(X_cascade_compras, y_test_compra)

print(f'Cascading Classifier - Precisió sobre el Conjunt de Prova: {test_accuracy_meta_cascade_compras:.4f}')

#Calcular r^2 para el modelo de compras de cascada
r2_cascade_compras = r2_score(y_test_compra, pred_meta_cascade_compras)
print('R² para el modelo de compras de cascada:', r2_cascade_compras)
#error cuadrático medio para el modelo de compras de cascada
mse_cascade_compras = mean_squared_error(y_test_compra, pred_meta_cascade_compras)
print('Error Cuadrático Medio para Compras de Cascada:', mse_cascade_compras)

#-------------------------------------------------------------------------------------------------------------------------------

# Guardar Todos los Modelos utilizando Pickle
import pickle

# Guardar el modelo de regresión lineal para ventas
with open('./Modelos/modelo_venta.pkl', 'wb') as file:
    pickle.dump(modelo_venta, file)

# Guardar el modelo de regresión lineal para compras
with open('./Modelos/modelo_compra.pkl', 'wb') as file:
    pickle.dump(modelo_compra, file)

# Guardar el modelo SVR para ventas
with open('./Modelos/modelo_venta_svr.pkl', 'wb') as file:
    pickle.dump(modelo_venta_svr, file)

# Guardar el modelo SVR para compras
with open('./Modelos/modelo_compra_svr.pkl', 'wb') as file:
    pickle.dump(modelo_compra_svr, file)

# Guardar el modelo de cascada ventas
with open('./Modelos/meta_classifier_cascade_Ventas.pkl', 'wb') as file:
    pickle.dump(meta_classifier_cascade, file)

# Guardar el modelo de cascada compras
with open('./Modelos/meta_classifier_cascade_Compras.pkl', 'wb') as file:
    pickle.dump(meta_classifier_cascade_compras, file)

# Crear tabla de datos de los resultados
resultados_r2 = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'SVR', 'Cascada'],
    'Ventas': [r2_score(y_test_venta, y_pred_venta), r2_venta_svr, r2_cascade],
    'Compras': [r2_score(y_test_compra, y_pred_compra), r2_compra_svr, r2_cascade_compras]
})

resultados_mse = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'SVR', 'Cascada'],
    'Ventas': [mean_squared_error(y_test_venta, y_pred_venta), mse_venta_svr, mse_cascade],
    'Compras': [mean_squared_error(y_test_compra, y_pred_compra), mse_compra_svr, mse_cascade_compras]
})

# Guardar los resultados en un archivo CSV
resultados_r2.to_csv('./resultados_modelos_r2.csv', index=False)
resultados_mse.to_csv('./resultados_modelos_mse.csv', index=False)

# Visualizar los resultados
print('Resultados de R² para los Modelos:')
print(resultados_r2)
print('\nResultados de Error Cuadrático Medio para los Modelos:')
print(resultados_mse)

# Graficos de lineas de predicciones por modelo de ventas
plt.figure(figsize=(12, 6))
plt.plot(y_test_venta.values, label='Real', color='blue')
plt.plot(y_pred_venta, label='Regresión Lineal', color='green')
# plt.plot(y_pred_venta_svr, label='SVR', color='red')
# plt.plot(pred_meta_cascade, label='Cascada', color='purple')
plt.title('Predicciones de Ventas por Modelo')
plt.legend()
# plt.show()