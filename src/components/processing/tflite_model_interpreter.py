import numpy as np
import cv2
import tensorflow as tf

class TFLiteModelInterpreter:
    """
    Clase para la interpretación de modelos TensorFlow Lite específicamente
    diseñada para el procesamiento de video. Inicializa el modelo TFLite y 
    permite la inferencia en frames de video.
    
    Attributes:
        interpreter (tf.lite.Interpreter): Intérprete de TensorFlow Lite.
        input_details (dict): Detalles del tensor de entrada del modelo.
        output_details (dict): Detalles del tensor de salida del modelo.
    """
    def __init__(self, model_path):
        """
        Inicializa el intérprete de TensorFlow Lite con el modelo especificado
        y prepara los tensores necesarios.
        
        Args:
            model_path (str): Ruta al archivo del modelo TensorFlow Lite.
        
        Raises:
            ValueError: Error si no se puede cargar el modelo.
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Error al cargar el modelo TensorFlow Lite {model_path}: {e}")
            raise ValueError("No se pudo cargar el modelo TFLite") from e

    def set_input_tensor(self, frame):
        """
        Preprocesa el frame y establece el tensor de entrada del modelo para su procesamiento.
        
        Args:
            frame (UMat): Frame de video a procesar.
        
        Raises:
            Exception: Error al establecer el tensor de entrada.
        """
        # Normalizar el frame de tal manera que sea valido para el tensor de entrada
        input_data = self.preprocess_frame(frame=frame)
        try:
            if input_data.shape != tuple(self.input_details[0]['shape']):
                raise ValueError(f"Error: Se esperaba {self.input_details[0]['shape']}.")
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        except Exception as e:
            print(f"Error al establecer el tensor de entrada: {e}")
            raise

    def invoke(self):
        """
        Realiza la inferencia utilizando el modelo cargado en el intérprete.
        
        Raises:
            Exception: Error durante la inferencia.
        """
        try:
            self.interpreter.invoke()
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            raise

    def get_output_tensor(self):
        """
        Obtiene el tensor de salida del modelo tras la inferencia.
        
        Returns:
            ndarray: Tensor de salida del modelo.
        
        Raises:
            Exception: Error al obtener el tensor de salida.
        """
        try:
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        except Exception as e:
            print(f"Error al obtener el tensor de salida: {e}")
            raise

    def preprocess_frame(self, frame, width = 320, height = 240):
        """
        Preprocesa un frame de video para adecuarlo a las especificaciones
        del modelo de Tensorflow Lite.
        
        Args:
            frame (UMat): Frame de video a procesar.
            width (int): Ancho deseado del frame para la entrada del modelo.
            height (int): Alto deseado del frame para la entrada del modelo.
        
        Returns:
            ndarray: Frame procesado y listo para ser usado como entrada en el modelo.
        """
        # Redimensionar el frame al tamaño de entrada esperado por el modelo
        img_resized = cv2.resize(frame, (width, height))
        # Normalizar los pixeles
        img_normalized = img_resized / 255.0
        # Expandir las dimensiones para adaptarse al formato de entrada del modelo (batch size)
        input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)
        return input_data
