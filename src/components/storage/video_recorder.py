from datetime import datetime
import cv2

class VideoRecorder:
    """
    Clase para gestionar la grabación de video utilizando OpenCV. Esta clase permite iniciar, 
    detener y escribir frames en un archivo de video con configuraciones específicas.

    Attributes:
        save_path (str): Ruta donde se guardará el archivo de video.
        frame_rate (float): Tasa de cuadros por segundo (FPS) del video.
        resolution (tuple): Resolución del video (ancho, alto).
        codec (str): Codec utilizado para comprimir el video.
        is_recording (bool): Estado de la grabación (True si está grabando, False si no).
        writer (cv2.VideoWriter): Objeto de OpenCV para escribir el video.
    """
    
    def __init__(self, save_path, frame_rate=10.0, resolution_option=2, codec='mp4v'):
        """
        Inicializa un objeto VideoRecorder con las configuraciones especificadas.

        Args:
            save_path (str): Ruta donde se guardará el archivo de video.
            frame_rate (float): Tasa de cuadros por segundo (FPS) del video. Por defecto es 10.0.
            resolution_option (int): Opción de resolución deseada (1, 2, 3 o 4).
            codec (str): Codec utilizado para comprimir el video. Por defecto es 'mp4v'.
        
        Raises:
            ValueError: Si la opción de resolución no es válida.
        """
        # Formato de fecha y hora para el nombre del archivo
        current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        self.save_path = f"{save_path}/{current_time}.mp4"
        self.frame_rate = frame_rate
        self.resolution = self.set_resolution(resolution_option=resolution_option)
        self.codec = codec
        self.is_recording = False
        self.writer = None
        
    def set_resolution(self, resolution_option):
        """
        Configura la resolución del video basada en la opción de resolución.

        Args:
            resolution_option (int): Opción de resolución deseada (1, 2, 3 o 4).

        Returns:
            tuple: Resolución del video (ancho, alto).

        Raises:
            ValueError: Si la opción de resolución no es válida.
        """
        resolutions = {
            1: (320*2, 240),
            2: (640*2, 480),
            3: (1280*2, 720),
            4: (1920*2, 1080)
        }
        if resolution_option not in resolutions:
            raise ValueError("Error: Resolución no válida. Las opciones son 1: '240p', 2: '480p', 3: '720p', 4: '1080p'.")
        return resolutions[resolution_option]

    def start_recording(self):
        """
        Inicia la grabación de video creando un objeto VideoWriter con las configuraciones especificadas.
        """
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.frame_rate, self.resolution)
        self.is_recording = True
        print(f"Grabación iniciada, guardando en {self.save_path}")

    def write_frame(self, frame):
        """
        Escribe un frame al video si la grabación está activa.

        Args:
            frame (ndarray): Frame de video a escribir.
        """
        if self.is_recording:
            self.writer.write(frame)

    def stop_recording(self):
        """
        Detiene la grabación de video y libera el objeto VideoWriter.
        """
        if self.is_recording:
            self.writer.release()
            print("Grabación detenida y archivo guardado.")
            self.is_recording = False
