from fastapi import FastAPI

from src.serving.schemas import GenerateRequest, GenerateResponse
from src.serving.worker import InferenceWorker

app = FastAPI(title="Mini LLM Runtime Engine")
worker = InferenceWorker(model_name="distilgpt2")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    out = worker.generate(
        request_id=req.request_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
    )
    return GenerateResponse(**out)