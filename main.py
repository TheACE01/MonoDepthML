from src.components.user_interface.depth_estimation_app import DepthEstimationApp

if __name__ == '__main__':
    # Inicializar la aplicacion
    app = DepthEstimationApp(
        "src/tensorflow_models/lite_models/monocular-depth-estimation2.0_fp16.tflite"
    )
    # Empezar con la ejecucion
    app.run()
