import cv2
import numpy as np

def preprocess_frame(frame, width = 320, height = 240):
    """Preprocesa un frame de video para la inferencia del modelo de Tensorflow Lite.
    
    Args:
        frame (UMat): _description_
        width (int): Ancho de imagen que necesita el interprete del modelo para realizar la inferencia.
        height (int): Alto de imagen que necesita el interprete del modelo para realizar la inferencia.
    """
    # Redimensionar el frame al tamano de entrada esperado por el modelo
    img_resized = cv2.resize(frame, (width, height))
    # Normalizar los pixeles
    img_normalized = img_resized / 255.0
    # Expandir las dimensiones para adaptarse al formato de entrada del modelo (batch size)
    input_data = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    return input_data