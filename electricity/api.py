"""FastAPI service that serves a saved Keras model.

Endpoints:
    GET /: welcome message.
    POST /predict: upload a ``.npy`` / ``.npz`` array, receive the flattened predictions.

The model is loaded once at import time from ``MODEL_PATH`` (environment variable,
default ``rnn_model.keras`` next to this file).
"""
import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

# Resolve model path relative to this file (works locally + Cloud Run)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "rnn_model.keras"))

# Load model once at startup
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

@app.get("/")
async def root():
    """Return a static welcome message (health check)."""
    return {"message": "welcome to our app"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run the model on an uploaded NumPy array.

    Args:
        file: ``.npy`` file, or ``.npz`` file whose first array is used. The array must
            have the input shape the loaded model expects, e.g.
            ``(n_samples, n_features)`` or ``(n_samples, timesteps, n_features)``.
            Pickled arrays are rejected (``allow_pickle=False``).

    Returns:
        ``{"input_shape": list[int], "prediction": list[float]}`` with predictions
        flattened to 1D, or HTTP 400 with ``{"error": str}`` on any failure.
    """
    try:
        contents = await file.read()

        # Load .npy or .npz safely
        X = np.load(io.BytesIO(contents), allow_pickle=False)
        if isinstance(X, np.lib.npyio.NpzFile):
            X = X[list(X.files)[0]]

        X = np.asarray(X)

        # Predict and flatten to 1D list
        y = model.predict(X).flatten().tolist()
        return {"input_shape": list(X.shape), "prediction": y}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("electricity.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
