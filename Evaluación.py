import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def remover_valores_atipicos(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

def main(datos_archivo, modelo_archivo):
    try:
        housing = pd.read_csv(datos_archivo)
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo '{datos_archivo}'")
        return


    #PROCESAMIENTO DE LOS DATOS
    housing = housing.filter(["price", "bed", "bath", "acre_lot", "house_size"])

    housing = housing.dropna()

    for col in ['bath', 'price', 'acre_lot', 'bed', 'house_size']:
        housing = remover_valores_atipicos(housing, col)

    columnas_caracteristicas = ["bed", "bath", "acre_lot", "house_size"]
    X = housing[columnas_caracteristicas]

    y = housing['price']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    #CARGA EL OBJETO DEL MODELO ENTRENADO
    try:
        modelo = joblib.load(modelo_archivo)
    except FileNotFoundError:
        print(f"No se pudo encontrar el archivo '{modelo_archivo}'")
        return

    print(f"Modelo cargado desde '{modelo_archivo}'")

    #PREDECIR PRECIOS DEL CONJUNTO DE EVALUACIÓN
    predicciones_test = modelo.predict(X_test)

    #EVALUAR EL RENDIMIENTO DEL MODELO
    mae_test = mean_absolute_error(y_test, predicciones_test)
    mse_test = mean_squared_error(y_test, predicciones_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, predicciones_test)

    print(f'\nMAE en conjunto de prueba: {mae_test:.2f}')
    print(f'MSE en conjunto de prueba: {mse_test:.2f}')
    print(f'RMSE en conjunto de prueba: {rmse_test:.2f}')
    print(f'R^2 en conjunto de prueba: {r2_test:.2f}')

    plt.figure(figsize=(12, 8))
    num_datos = 100
    plt.scatter(y_test[:num_datos], predicciones_test[:num_datos], c='b', alpha=0.5, label='Predicciones')
    plt.plot(y_test[:num_datos], y_test[:num_datos], color='r', label='Valor Real')
    plt.xlabel("Valor Real")
    plt.ylabel("Predicción")
    plt.title("Valores Reales vs Predicciones (Conjunto de Prueba)")
    plt.legend()
    plt.show()

    #REALIZAR PREDICCIONES CON DATOS NUEVOS
    ejemplos_prediccion = [
        [3, 2.5, 0.3, 1800],
        [4, 3.5, 0.45, 3200],
        [5, 4, 0.72, 3600]
    ]

    precios_reales = [275000, 720000, 489000]

    ejemplos_prediccion_df = pd.DataFrame(ejemplos_prediccion, columns=columnas_caracteristicas)
    ejemplos_prediccion_scaled = scaler.transform(ejemplos_prediccion_df)
    predicciones_ejemplos = modelo.predict(ejemplos_prediccion_scaled)

    ejemplos_prediccion_df = pd.DataFrame(ejemplos_prediccion, columns=columnas_caracteristicas)
    ejemplos_prediccion_scaled = scaler.transform(ejemplos_prediccion_df)
    predicciones_ejemplos = modelo.predict(ejemplos_prediccion_scaled)

    print("\nPredicciones para ejemplos específicos:")
    for i, ejemplo in enumerate(ejemplos_prediccion):
        caracteristicas_str = ', '.join(f'{col}: {val}' for col, val in zip(columnas_caracteristicas, ejemplo))
        precio_real = precios_reales[i]
        prediccion = predicciones_ejemplos[i]
        diferencia = abs(prediccion - precio_real)

        print(f"Ejemplo {i + 1}: Características ({caracteristicas_str})")
        print(f"Precio Real: {precio_real}")
        print(f"Predicción: {prediccion:.2f}")
        print(f"Diferencia: {diferencia:.2f}\n")

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, 4), precios_reales, marker='o', linestyle='-', color='r', label='Valor Real')
    plt.plot(range(1, 4), predicciones_ejemplos, marker='o', linestyle='--', color='b', label='Predicción')
    plt.xlabel("Ejemplo")
    plt.ylabel("Precio")
    plt.title("Predicción vs Valor Real para Ejemplos Específicos")
    plt.xticks(range(1, 4), ['Ejemplo 1', 'Ejemplo 2', 'Ejemplo 3'])
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento y evaluación de modelo de regresión')
    parser.add_argument('datos_archivo', type=str, help='Nombre del archivo CSV de datos')
    parser.add_argument('modelo_archivo', type=str, help='Nombre del archivo del modelo entrenado (pkl)')
    args = parser.parse_args()

    main(args.datos_archivo, args.modelo_archivo)
