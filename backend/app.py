"""
Aplicación Flask para Predicción de Calidad del Aire (PM2.5)
Parte 2: Despliegue Web del Modelo - Evaluación Parcial 2
Alumno: Orrantia Gonzalez German Sisac
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Permitir solicitudes desde el frontend

# Cargar el modelo y preprocesadores al iniciar la aplicación
print("Cargando modelo y preprocesadores...")
modelo = joblib.load('models/mejor_modelo.pkl')
scaler = joblib.load('models/scaler.pkl')
features_info = joblib.load('models/features_info.pkl')
print("Modelo y preprocesadores cargados exitosamente")

# Obtener las características que espera el modelo
FEATURES_NUMERICAS = ['so2', 'no2', 'rspm', 'spm']
ALL_FEATURES = features_info.get('features_numericas', FEATURES_NUMERICAS)

@app.route('/')
def home():
    """Ruta principal - sirve el archivo HTML"""
    # Servir el index.html desde la carpeta raíz del proyecto
    return send_from_directory('..', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones
    Espera un JSON con los valores de los contaminantes:
    {
        "so2": float,
        "no2": float,
        "rspm": float,
        "spm": float
    }
    """
    try:
        # Obtener datos del request
        data = request.get_json()

        # Validar que todos los campos necesarios estén presentes
        campos_faltantes = []
        for feature in FEATURES_NUMERICAS:
            if feature not in data:
                campos_faltantes.append(feature)

        if campos_faltantes:
            return jsonify({
                'error': f'Campos faltantes: {", ".join(campos_faltantes)}'
            }), 400

        # Extraer valores y convertir a float
        try:
            valores = [float(data[feature]) for feature in FEATURES_NUMERICAS]
        except ValueError as e:
            return jsonify({
                'error': 'Los valores deben ser numeros validos'
            }), 400

        # Validar que los valores sean positivos
        if any(v < 0 for v in valores):
            return jsonify({
                'error': 'Los valores de los contaminantes no pueden ser negativos'
            }), 400

        # Escalar solo las variables numéricas
        X_num = np.array([valores])
        X_num_scaled = scaler.transform(X_num)

        # Concatenar las categóricas (en cero) después del escalado
        FEATURES_CATEGORICAS = features_info.get('features_categoricas', [])
        X_cat = np.zeros((1, len(FEATURES_CATEGORICAS)))
        X_final = np.concatenate([X_num_scaled, X_cat], axis=1)

        # Realizar la predicción
        prediccion = modelo.predict(X_final)[0]

        # Determinar la categoría de calidad del aire basada en PM2.5
        categoria, descripcion, color = obtener_categoria_aqi(prediccion)

        # Preparar respuesta
        respuesta = {
            'prediccion': round(float(prediccion), 2),
            'categoria': categoria,
            'descripcion': descripcion,
            'color': color,
            'valores_entrada': {
                'SO2': valores[0],
                'NO2': valores[1],
                'RSPM': valores[2],
                'SPM': valores[3]
            }
        }

        return jsonify(respuesta)

    except Exception as e:
        return jsonify({
            'error': f'Error al procesar la prediccion: {str(e)}'
        }), 500

def obtener_categoria_aqi(pm25_valor):
    """
    Clasifica el valor de PM2.5 en categorías de calidad del aire
    Basado en estándares internacionales de AQI
    """
    if pm25_valor <= 12:
        return "Buena", "La calidad del aire es satisfactoria.", "#00e400"
    elif pm25_valor <= 35.4:
        return "Moderada", "La calidad del aire es aceptable.", "#ffff00"
    elif pm25_valor <= 55.4:
        return "Dañina para grupos sensibles", "Personas sensibles pueden experimentar efectos.", "#ff7e00"
    elif pm25_valor <= 150.4:
        return "Dañina", "Todos pueden comenzar a experimentar efectos en la salud.", "#ff0000"
    elif pm25_valor <= 250.4:
        return "Muy Dañina", "Alerta de salud: todos pueden experimentar efectos graves.", "#8f3f97"
    else:
        return "Peligrosa", "Emergencia sanitaria: toda la población puede verse afectada.", "#7e0023"

@app.route('/info', methods=['GET'])
def info():
    """
    Endpoint para obtener información sobre el modelo
    """
    info_modelo = {
        'modelo': 'k-NN Optimizado (GridSearchCV)',
        'algoritmo': 'K-Nearest Neighbors Regressor',
        'features_entrada': FEATURES_NUMERICAS,
        'target': features_info['target'],
        'descripcion': 'Modelo entrenado para predecir niveles de PM2.5 basándose en otros contaminantes atmosféricos',
        'metricas': {
            'nota': 'Métricas exactas se encuentran en el notebook de entrenamiento'
        }
    }
    return jsonify(info_modelo)

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servidor está funcionando"""
    return jsonify({
        'status': 'OK',
        'mensaje': 'Servidor Flask funcionando correctamente'
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Servidor de Prediccion de Calidad del Aire")
    print("="*60)
    print(f"Modelo: k-NN Optimizado")
    print(f"Features de entrada: {', '.join(FEATURES_NUMERICAS)}")
    print(f"Target: {features_info.get('target', 'pm2_5')}")
    print("="*60)
    print("\nIniciando servidor en http://localhost:5000")
    print("Abre tu navegador en: http://localhost:5000")
    print("\nPresiona Ctrl+C para detener el servidor\n")
    
    # Ejecutar la aplicación
    app.run(debug=True, host='0.0.0.0', port=5000)
