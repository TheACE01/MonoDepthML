from src.components.processing.tflite_model_interpreter import TFLiteModelInterpreter
from src.components.processing.camera_manager import CameraManager
from src.components.storage.video_recorder import VideoRecorder
from src.components.processing.video_processor import VideoProcessor

class DepthEstimationApp:
    """
    Aplicación para la estimación de profundidad utilizando un modelo TFLite, captura de video 
    desde una cámara web y procesamiento y almacenamiento de video.

    Attributes:
        resolution_option (int): Opción de resolución de la cámara.
        depth_model (TFLiteModelInterpreter): Intérprete del modelo TFLite para la estimación de profundidad.
        camera_manager (CameraManager): Gestor de la cámara web.
        video_recorder (VideoRecorder): Grabador de video.
        video_processor (VideoProcessor): Procesador de video.
        enable_storage (bool): Indicador de si la grabación de video está habilitada.
    """
    
    def __init__(self, tflite_model_path):
        """
        Inicializa un objeto DepthEstimationApp con el modelo TFLite especificado.

        Args:
            tflite_model_path (str): Ruta al archivo del modelo TFLite.
        """
        self.resolution_option = 2
        self.depth_model = TFLiteModelInterpreter(model_path=tflite_model_path)
        self.camera_manager = None
        self.video_recorder = None
        self.video_processor = VideoProcessor()
        self.enable_storage = False

    def run(self):
        """
        Ejecuta la aplicación de estimación de profundidad. Captura video de la cámara,
        realiza la estimación de profundidad, procesa el video y opcionalmente lo graba.
        """
        # Seleccionar la resolución de la cámara web
        self.ask_for_frame_resolution()
        # Habilitar o deshabilitar grabación de video
        self.ask_for_video_recording()
        if self.enable_storage:
            self.video_recorder = VideoRecorder(
                save_path="src/videos",
                frame_rate=10.0,
                resolution_option=self.resolution_option,
                codec='mp4v')
            self.video_recorder.start_recording()

        try:
            print("Iniciando estimación de fondo monocular")
            while True:
                frame = self.camera_manager.get_frame()
                self.depth_model.set_input_tensor(frame=frame)
                self.depth_model.invoke()
                output_data = self.depth_model.get_output_tensor()
                self.video_processor.normalize_output(output_data=output_data, frame=frame)
                self.video_processor.calculate_fps()
                self.video_processor.visualize()
                if self.enable_storage:
                    self.video_recorder.write_frame(frame=self.video_processor.get_output())
                if self.video_processor.validate_stop():
                    break
        finally:
            self.cleanup()
            
    def ask_for_frame_resolution(self):
        """
        Solicita al usuario que seleccione la resolución de la cámara y configura la cámara 
        en consecuencia.
        """
        prompt_message = "\nSeleccionar resolución de la cámara:\n1 - 240p\n2 - 480p\n3 - 720p\n4 - 1080p\n"
        while True:
            choice = input(prompt_message)
            if choice in ('1', '2', '3', '4'):
                break
        # Configurar la resolución del administrador de cámara web
        self.resolution_option = int(choice)
        self.camera_manager = CameraManager(source=0, resolution_option=self.resolution_option)
        
    def ask_for_video_recording(self):
        """
        Solicita al usuario que habilite o deshabilite la grabación de video.
        """
        prompt_message = "\nHabilitar grabación de video:\n1 - Si\n2 - No\n"
        while True:
            choice = input(prompt_message)
            if choice in ('1', '2'):
                break
        if choice == '1':
            self.enable_storage = True

    def cleanup(self):
        """
        Libera los recursos utilizados por la aplicación, como la cámara y el grabador de video.
        """
        self.camera_manager.release()
        if self.enable_storage:
            self.video_recorder.stop_recording()
        self.video_processor.release()
        print("La estimación de fondo monocular ha finalizado")
