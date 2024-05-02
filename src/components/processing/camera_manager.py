import cv2

class CameraManager:
    """
    Gestiona la conexión y captura de video desde una cámara utilizando OpenCV. Esta clase permite
    abrir una fuente de video, leer frames y cerrar la conexión de manera controlada.

    Attributes:
        cap (cv2.VideoCapture): Objeto de captura de video de OpenCV que gestiona la transmisión de
        la cámara.
    """

    def __init__(self, source=0):
        """
        Inicializa un objeto CameraManager que intenta abrir la fuente de video especificada.
        
        Args:
            source (int or str): El índice del dispositivo de la cámara o la ruta del video a abrir.
            Por defecto, se usa 0, que generalmente se refiere a la cámara web principal
            del sistema.

        Raises:
            Exception: Si no se puede acceder a la fuente de video especificada.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Error: No se pudo acceder a la cámara web.")

    def get_frame(self):
        """
        Lee el siguiente frame de la fuente de video.

        Returns:
            ndarray: El frame capturado de la cámara.

        Raises:
            Exception: Si no se puede leer un frame de la fuente de video.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Error: No se pudo obtener el frame.")
        return frame

    def release(self):
        """
        Libera la fuente de video, cerrando la conexión con la cámara y liberando los recursos
        asociados.
        """
        self.cap.release()
