import cv2
from datetime import datetime

class VideoRecorder:
    def __init__(self, save_path, frame_rate=20.0, resolution=(640, 480), codec='mp4v'):
        """
        Inicializa el VideoRecorder con la ruta del archivo y configuraciones.
        """
        # Formato de fecha y hora para el nombre del archivo
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = f"{save_path}/{current_time}.mp4"
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.codec = codec
        self.is_recording = False
        self.writer = None

    def start_recording(self):
        """
        Inicia la grabación del video.
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.frame_rate, self.resolution)
        self.is_recording = True
        print(f"Grabación iniciada, guardando en {self.save_path}")

    def write_frame(self, frame):
        """
        Escribe un frame al video si la grabación está activa.
        """
        if self.is_recording:
            self.writer.write(frame)

    def stop_recording(self):
        """
        Detiene la grabación y libera los recursos.
        """
        if self.is_recording:
            self.writer.release()
            print("Grabación detenida y archivo guardado.")
            self.is_recording = False
