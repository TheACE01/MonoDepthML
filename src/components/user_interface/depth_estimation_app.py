from src.components.processing.tflite_model_interpreter import TFLiteModelInterpreter
from src.components.processing.camera_manager import CameraManager
from src.components.storage.video_recorder import VideoRecorder
from src.components.processing.video_processor import VideoProcessor

class DepthEstimationApp:
    """
    Aplicación para realizar la estimación de profundidad en tiempo real utilizando
    un modelo TFLite.
    Integra componentes para la captura de video, procesamiento de frames, y grabación
    opcional de resultados.

    Attributes:
        depth_model (TFLiteModelInterpreter): Intérprete para el modelo TFLite.
        camera_manager (CameraManager): Gestor de la cámara para captura de video.
        video_recorder (VideoRecorder): Gestor de grabación de video (opcional).
        video_processor (VideoProcessor): Procesador de video para visualización de resultados.
        enable_storage (bool): Indicador de si se habilita la grabación de video.
    """
    def __init__(self, tflite_model_path):
        """
        Inicializa la aplicación con los componentes necesarios para la estimación de profundidad.

        Args:
            tflite_model_path (str): Ruta al modelo TFLite utilizado para la inferencia.
        """
        self.depth_model = TFLiteModelInterpreter(model_path=tflite_model_path)
        self.camera_manager = CameraManager(source=0)
        self.video_recorder = None
        self.video_processor = VideoProcessor()
        self.enable_storage = False

    def run(self):
        """
        Inicia el ciclo principal de la aplicación, gestionando la captura y procesamiento de video,
        así como la interacción con el usuario para controlar la grabación de resultados.
        """
        self.ask_for_video_recording()
        if self.enable_storage:
            self.video_recorder = VideoRecorder(
                save_path="src/videos",
                frame_rate=20.0,
                resolution=(1280, 480),
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

    def ask_for_video_recording(self):
        """
        Interactúa con el usuario para determinar si desea habilitar la grabación de los resultados
        de video.
        """
        prompt_message = "\nElige una opción:\n1 - Guardar resultados\n2 - No guardar resultados\n"
        while True:
            choice = input(prompt_message)
            if choice in ('1', '2'):
                break
        if choice == '1':
            self.enable_storage = True

    def cleanup(self):
        """
        Limpia los recursos utilizados por la aplicación, asegurando que todos los componentes sean
        liberados
        adecuadamente al terminar la ejecución.
        """
        self.camera_manager.release()
        if self.enable_storage:
            self.video_recorder.stop_recording()
        self.video_processor.release()
