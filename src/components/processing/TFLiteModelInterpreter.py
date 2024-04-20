import numpy as np
import tensorflow as tf

class TFLiteModelInterpreter:
    def __init__(self, model_path):
        """Inicializa el intérprete de TensorFlow Lite con el modelo especificado."""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except Exception as e:
            print(f"Error al cargar el modelo TensorFlow Lite {model_path}: {e}")
            raise ValueError("No se pudo cargar el modelo TFLite.")

    def set_input_tensor(self, input_data):
        """Establece el tensor de entrada del modelo."""
        try:
            if input_data.shape != tuple(self.input_details[0]['shape']):
                raise ValueError(f"Forma de datos de entrada incorrecta, se esperaba {self.input_details[0]['shape']}, pero se recibió {input_data.shape}.")
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        except Exception as e:
            print(f"Error al establecer el tensor de entrada: {e}")
            raise

    def invoke(self):
        """Realiza la inferencia en el modelo."""
        try:
            self.interpreter.invoke()
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            raise

    def get_output_tensor(self):
        """Obtiene el tensor de salida del modelo."""
        try:
            return self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        except Exception as e:
            print(f"Error al obtener el tensor de salida: {e}")
            raise