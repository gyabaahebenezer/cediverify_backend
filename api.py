import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

from inference import CurrencyVerifier

# Set this as an environment variable on your hosting platform. Without it,
# the endpoint is open to anyone -- fine for local testing, not for a
# deployed app that costs you compute per request.
API_KEY = os.environ.get("API_KEY")

verifier = None  # set once at startup, reused across every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    global verifier
    verifier = CurrencyVerifier()
    verifier.preload_all()   # load every autoencoder once here, not per-request
    print("Models loaded, ready to serve.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your app's actual origin before production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB -- adjust to whatever your app actually sends
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify")
async def verify(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = verifier.verify(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)