import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Pedir al usuario que ingrese el nombre del archivo CSV de datos
datos_archivo = input("Ingrese el nombre del archivo CSV de datos: ")

# Cargar los datos desde el archivo CSV proporcionado por el usuario
try:
    housing = pd.read_csv(datos_archivo)
except FileNotFoundError:
    print(f"No se pudo encontrar el archivo '{datos_archivo}'")
    exit(1)

# Filtrar las columnas relevantes
housing = housing.filter(["price", "bed", "mean", "bath", "acre_lot", "house_size"])

# Eliminar filas con valores faltantes
housing = housing.dropna()

# Función para remover valores atípicos basados en IQR
def remover_valores_atipicos(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

# Remover valores atípicos en varias columnas
for col in ['bath', 'price', 'acre_lot', 'bed', 'house_size']:
    housing = remover_valores_atipicos(housing, col)

# Definir variable objetivo
y = housing['price']

# Seleccionar columnas de características
columnas_caracteristicas = ["bed", "bath", "acre_lot", "house_size"]
X = housing[columnas_caracteristicas]

# Imputar valores faltantes en las columnas de características
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Escalar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Pedir al usuario que ingrese el nombre del archivo del modelo entrenado
modelo_archivo = input("Ingrese el nombre del archivo del modelo entrenado (pkl): ")

# Cargar el modelo entrenado desde el archivo proporcionado por el usuario
try:
    modelo = joblib.load(modelo_archivo)
except FileNotFoundError:
    print(f"No se pudo encontrar el archivo '{modelo_archivo}'")
    exit(1)

print(f"Modelo cargado desde '{modelo_archivo}'")

# Realizar predicciones en el conjunto de prueba
predicciones_test = modelo.predict(X_test)

# Calcular las métricas de evaluación: MAE, MSE, RMSE
mae_test = mean_absolute_error(y_test, predicciones_test)
mse_test = mean_squared_error(y_test, predicciones_test)
rmse_test = np.sqrt(mse_test)

# Mostrar las métricas de evaluación
print(f'MAE en conjunto de prueba: {mae_test:.2f}')
print(f'MSE en conjunto de prueba: {mse_test:.2f}')
print(f'RMSE en conjunto de prueba: {rmse_test:.2f}')

# Visualizar valores reales versus predicciones con colores distintos
plt.figure(figsize=(12, 8))
plt.scatter(y_test, predicciones_test, c='b', alpha=0.5, label='Predicciones')
plt.plot(y_test, y_test, color='r', label='Valor Real')
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Valores Reales vs Predicciones (Conjunto de Prueba)")
plt.legend()
plt.show()

# Ejemplos de predicción sobre nuevos datos (solo como demostración)
nuevos_datos = np.array([[1000, 3, 2, 800],
                         [1500, 4, 3, 1200],
                         [800, 2, 1, 600]])

# Realizar predicciones sobre nuevos datos con el modelo cargado
predicciones_nuevos = modelo.predict(nuevos_datos)

print("Predicciones sobre nuevos datos (solo para demostración):")
for i, pred in enumerate(predicciones_nuevos):
    print(f'Ejemplo {i+1}: ${pred:.2f}')
