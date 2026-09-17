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
    return {"message": "welcome to our app"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
