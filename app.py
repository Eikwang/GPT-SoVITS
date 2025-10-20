import os
import sys
import threading
from GPT_SoVITS.inference_webui_fast import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

if __name__ == "__main__":
    event = threading.Event()

    fastapi_thread = threading.Thread(target=start_service)
    gradio_thread = threading.Thread(target=start_gradio)

    fastapi_thread.start()
    gradio_thread.start()

    fastapi_thread.join()
    gradio_thread.join()