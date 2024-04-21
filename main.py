from src.components.user_interface.user_interface import start_depth_estimation

start_depth_estimation(
    "src/tensorflow_models/lite_models/monocular-depth-estimation3.0_fp16.tflite",
    "src/videos")
