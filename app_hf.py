import os
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.wsgi import WSGIMiddleware
from api import app as flask_app

# Wrap our Flask API (which serves frontend/index.html & /predict) into ASGI
app = Starlette()
app.mount("/", WSGIMiddleware(flask_app))

# Optional Gradio mount to satisfy Gradio healthchecks
try:
    import gradio as gr
    with gr.Blocks(title="ECG Arrhythmia AI") as demo:
        gr.Markdown("# ECG Arrhythmia AI Dashboard")
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except Exception as e:
    print(f"Gradio mount notice: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Clinical ECG Web Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
