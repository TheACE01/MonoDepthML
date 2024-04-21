import cv2
import time
import sys
import numpy as np
from src.components.processing.TFLiteModelInterpreter import TFLiteModelInterpreter
from src.components.validation.format_validator import preprocess_frame
from src.components.storage.video_recorder import VideoRecorder

def start_depth_estimation(model_path, save_path):
    print("ESTIMACIÓN DE FONDO MONOCULAR EN TIEMPO REAL")
    # Preguntarle al usuario si desea iniciar el programa
    start = input("¿Deseas iniciar la ejecución? (s/n): ")
    if start.lower() != 's':
        print("Saliendo del programa.")
        return
    # Instanciar la clase del interprete del modelo de Tensorflow Lite
    try:
        depth_model_interpreter = TFLiteModelInterpreter(model_path=model_path)
    except ValueError as ve:
        print(ve)
        return
    except Exception as e:
        print(f"Se encontro un error inesperado: {e}")
        return
    # Preguntarle al usuario si desea o no alcenar los resultados de los videos
    enable_storage = False
    choice = input("Elige una opción:\n1 - Guardar resultados\n2 - No guardar resultados\n")
    if choice == '1':
        print("\nLos resultados de las predicciones seran almacenados en src/results")
        # Habilitar el almacenamiento de video
        enable_storage = True
        recorder = VideoRecorder(save_path=save_path, frame_rate=20.0, resolution=(640 * 2, 480), codec='mp4v')
        recorder.start_recording()
    elif choice == '2':
        print("\nLos resultados de las predicciones no serán almacenados en el dispositivo.")
    else:
        print("Opción no válida. No se guardarán los resultados.")
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara web.")
        return
    # Ciclo principal del programa
    try:
        prev_frame_time = time.time()
        print("Iniciando estimacion de fondo monocular...")
        while True:
            # Leer el flujo de video proveniente de la camara web
            ret, frame = cap.read()
            # Finalizar la ejecucion en caso de que no se haya podido obtner el frame
            if not ret:
                print("Error: No se pudo obtener el frame.")
                break
            # Preprocesar el frame capturado
            input_data = preprocess_frame(frame)
            # Preparar el tensor de entrada con el frame preprocesado
            depth_model_interpreter.set_input_tensor(input_data=input_data)
            # Realizar la inferencia de estimacion de fondo monocular usando la instancia del interprete
            depth_model_interpreter.invoke()
            # Almacenar el resultado de la inferencia
            output_data = depth_model_interpreter.get_output_tensor()
            # Redimensionar la imagen de salida a la resolucion de la captura
            output_resized = cv2.resize(output_data, (frame.shape[1], frame.shape[0]))
            output_normalized = cv2.normalize(output_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            colored_output = cv2.applyColorMap(np.uint8(output_normalized), cv2.COLORMAP_BONE)
            # Calcular los FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {int(fps)}"
            # Combinar el frame original con la prediccion
            combined_output = np.hstack((frame, colored_output))
            # Mostrar el frame con las predicciones
            cv2.putText(combined_output, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Visualizador', combined_output)
            # Guardar el video si es necesario
            if enable_storage:
                recorder.write_frame(combined_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Detener la grabacion de video una vez finalizada la ejecucion
        if enable_storage:
            recorder.stop_recording()