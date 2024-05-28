# Estimación de Fondo Monocular

Este proyecto implementa una aplicación de estimación de fondo monocular utilizando un modelo TFLite, captura de video desde una cámara web y procesamiento y almacenamiento de video. La aplicación permite seleccionar la resolución de la cámara, habilitar o deshabilitar la grabación de video, y visualizar la estimación de fondo en tiempo real.

## Requisitos

- Python 3.9 o superior
- OpenCV
- TensorFlow
- Matplotlib

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/TheACE01/MonoDepthML.git
   cd MonoDepthML
   ```

2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Asegúrate de tener una cámara web conectada.

2. Ejecuta la aplicación:
   ```bash
   python main.py
   ```

3. Selecciona la resolución de la cámara:
   ```
   Seleccionar resolución de la cámara:
   1 - 240p
   2 - 480p
   3 - 720p
   4 - 1080p
   ```

4. Habilita o deshabilita la grabación de video:
   ```
   Habilitar grabación de video:
   1 - Si
   2 - No
   ```

5. La aplicación comenzará a capturar video, realizar la estimación de fondo, y mostrar el resultado en tiempo real. Presiona 'q' para detener la aplicación.

## Estructura del Proyecto

```plaintext
MonoDepthML/
│
├── src/
│   ├── components/
│   │   ├── processing/
│   │   │   ├── camera_manager.py
│   │   │   ├── tflite_model_interpreter.py
│   │   │   ├── video_processor.py
│   │   ├── storage/
│   │   │   ├── video_recorder.py
│   │   ├── user_interface/
│   │   │   ├── depth_estimation_app.py
│   ├── examples/
│   ├── tensorflow_models/
│   │   ├── lite_models/
│   │   │   ├── monocular-depth-estimation2.0
│   │   │   ├── monocular-depth-estimation3.0
│   │   ├── SavedModels/
│   ├── videos/
│   │   ├── 2024-05-26_16.54.56.mp4
│
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
├── unet.ipynb
```

## Descripción de Componentes

- `camera_manager.py`: Gestiona la conexión y captura de video desde la cámara web.
- `tflite_model_interpreter.py`: Interpreta el modelo TFLite para la estimación de fondo.
- `video_processor.py`: Procesa y visualiza el video en tiempo real, aplicando normalización y un mapa de colores.
- `video_recorder.py`: Gestiona la grabación y almacenamiento del video.
- `depth_estimation_app.py`: Contiene la lógica principal de la aplicación de estimación de fondo monocular.
- `main.py`: Punto de entrada principal de la aplicación.
- `unet.ipynb`: Notebook usado para entrenar y convertir el modelo para la estimación de fondo.
