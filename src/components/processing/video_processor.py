import time
import numpy as np
import cv2

class VideoProcessor:
    """
    Clase encargada de procesar y visualizar video en tiempo real, aplicando normalización y un 
    mapa de colores para visualizar la estimación de profundidad u otros resultados de inferencia.
    
    Attributes:
        color_map (int): Referencia al mapa de colores de OpenCV utilizado para visualizar
        la salida.
        prev_frame_time (float): Marca de tiempo del último frame procesado, utilizada para
        calcular FPS.
        frame_time_text (str): Texto para mostrar los FPS en la visualización.
        output (ndarray): La imagen resultante después de combinar el frame original con la
        salida procesada.
    """
    def __init__(self):
        self.color_map = cv2.COLORMAP_MAGMA
        self.prev_frame_time = time.time()
        self.frame_time_text = "FPS: ?"
        self.output = None

    def normalize_output(self, output_data, frame):
        """
        Procesa y combina el output de un modelo con el frame original, aplicando normalización 
        y un mapa de colores.
        
        Args:
            output_data (ndarray): Datos de salida del modelo de inferencia.
            frame (ndarray): Frame original capturado de la cámara.
        """
        # Redimensionar la imagen a el tamaño original de la captura de video
        output_resized = cv2.resize(output_data, (frame.shape[1], frame.shape[0]))
        # Normalizar los pixeles
        output_normalized = cv2.normalize(
            output_resized,
            None, alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX)
        # Aplicar un ColorMap para visualizar mejor la estimación
        colored_output = cv2.applyColorMap(np.uint8(output_normalized), self.color_map)
        # Combinar el frame original con la predicción
        self.output = np.hstack((frame, colored_output))

    def calculate_fps(self):
        """
        Calcula los frames por segundo (FPS) y actualiza el texto de visualización de los FPS.
        """
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        self.frame_time_text = f"FPS: {int(fps)}"

    def visualize(self):
        """
        Visualiza el frame combinado y los FPS sobre una ventana de OpenCV.
        """
        cv2.putText(
            self.output,
            self.frame_time_text,
            (7, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (100, 255, 0),
            3,
            cv2.LINE_AA)
        cv2.imshow('Visualizador', self.output)

    def get_output(self):
        """
        Devuelve la salida actual procesada y combinada.

        Returns:
            ndarray: El frame combinado actual.
        """
        return self.output

    def validate_stop(self):
        """
        Verifica si el usuario ha solicitado cerrar la aplicación.

        Returns:
            bool: True si el usuario ha presionado 'q', False en caso contrario.
        """
        return cv2.waitKey(1) & 0xFF == ord('q')

    def release(self):
        """
        Libera todos los recursos de OpenCV y cierra todas las ventanas.
        """
        cv2.destroyAllWindows()
