import cv2

class CameraManager:
    """
    Gestiona la conexión y captura de video desde una cámara utilizando OpenCV. Esta clase permite
    abrir una fuente de video, leer frames y cerrar la conexión de manera controlada, además de 
    configurar la resolución de la cámara.

    Attributes:
        cap (cv2.VideoCapture): Objeto de captura de video de OpenCV que gestiona la transmisión de
        la cámara.
    """

    def __init__(self, source=0, resolution_option=2):
        """
        Inicializa un objeto CameraManager que intenta abrir la fuente de video especificada y configura la resolución.

        Args:
            source (int or str): El índice del dispositivo de la cámara o la ruta del video a abrir.
                                 Por defecto, se usa 0, que generalmente se refiere a la cámara web principal
                                 del sistema.
            resolution_option (int): La opción de resolución deseada para la cámara. Puede ser 1, 2, 3 o 4.

        Raises:
            ValueError: Si no se puede acceder a la fuente de video especificada o si la resolución no es válida.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Error: No se pudo acceder a la cámara web.")

        self.set_resolution(resolution_option)

    def set_resolution(self, resolution_option):
        """
        Configura la resolución de la cámara y valida si la cámara soporta la resolución seleccionada.

        Args:
            resolution_option (int): La opción de resolución deseada para la cámara. Puede ser 1, 2, 3 o 4.

        Raises:
            ValueError: Si la resolución no es válida o si la cámara no soporta la resolución seleccionada.
        """
        resolutions = {
            1: (320, 240),
            2: (640, 480),
            3: (1280, 720),
            4: (1920, 1080)
        }

        if resolution_option not in resolutions:
            raise ValueError("Error: Resolución no válida. Las opciones son 1: '240p', 2: '480p', 3: '720p', 4: '1080p'.")

        width, height = resolutions[resolution_option]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Validar si la cámara soporta la resolución seleccionada
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if (actual_width, actual_height) != (width, height):
            raise ValueError(f"Error: La cámara no soporta la resolución seleccionada ({width}x{height})."
                             f" Resolución actual: ({actual_width}x{actual_height}).")

    def get_frame(self):
        """
        Lee el siguiente frame de la fuente de video.

        Returns:
            ndarray: El frame capturado de la cámara.

        Raises:
            ValueError: Si no se puede leer un frame de la fuente de video.
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
