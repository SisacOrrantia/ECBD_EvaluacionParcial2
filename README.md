# Predictor de Calidad del Aire PM2.5

Aplicación web para predecir niveles de PM2.5 utilizando un modelo de Machine Learning (k-Nearest Neighbors).

## Requisitos Previos

Para ejecutar este proyecto necesitas tener instalado:

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

Adicionalmente, debes generar los archivos de modelo ejecutando el notebook antes de iniciar la aplicación.

## Instalación

### 1. Instalar dependencias de Python

Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
cd backend
pip install -r requirements.txt
```

Esto instalará:
- Flask
- flask-cors
- joblib
- numpy
- scikit-learn
- pandas

### 2. Generar los modelos

Antes de iniciar la aplicación, debes generar los archivos del modelo ejecutando las celdas de la Parte 2 del notebook UTT ECBD Examen2.ipynb.

Estas celdas crearán la carpeta backend/models/ con los archivos:
- mejor_modelo.pkl (Modelo k-NN entrenado)
- scaler.pkl (Normalizador de datos)
- features_info.pkl (Información de características)

## Como Iniciar la Aplicación

### 1. Iniciar el servidor backend

Desde la carpeta del proyecto:

```bash
cd backend
python app.py
```

Verás un mensaje indicando que el servidor está corriendo en http://localhost:5000

### 2. Abrir la interfaz web

Abre el archivo index.html en tu navegador web, o accede directamente a http://localhost:5000

### 3. Usar la aplicación

Ingresa los valores de contaminantes en microgramos por metro cúbico:
- SO2 (Dióxido de azufre)
- NO2 (Dióxido de nitrógeno)
- RSPM (Material particulado respirable)
- SPM (Material particulado suspendido)

Haz clic en "Predecir PM2.5" y obtendrás la predicción con su categoría de calidad del aire.

## Información del Modelo

- Algoritmo: k-Nearest Neighbors (k-NN)
- Valor de k: 11 (optimizado mediante GridSearchCV)
- Variables de entrada: SO2, NO2, RSPM, SPM
- Variable objetivo: PM2.5
- Preprocesamiento: StandardScaler

## Solución de Problemas

### El servidor no inicia

Error: ModuleNotFoundError: No module named 'flask'

Solución: Instala las dependencias con pip install -r requirements.txt dentro de la carpeta backend

---

Error: FileNotFoundError: No such file or directory: 'models/mejor_modelo.pkl'

Solución: Ejecuta las celdas del notebook para generar los archivos .pkl del modelo en la carpeta backend/models/

### La página no carga predicciones

Error: Failed to fetch o error de CORS

Solución: Verifica que el servidor backend esté corriendo en http://localhost:5000 y que no haya otro proceso usando ese puerto

### Predicciones incorrectas o error de validación

Solución: Verifica que los valores ingresados sean números positivos en el rango adecuado (valores típicos entre 0 y 500 para contaminantes atmosféricos)
