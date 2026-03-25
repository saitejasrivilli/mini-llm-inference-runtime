import torch

from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine
from src.core.metrics import summarize_states


def _ensure_qengine():
    supported = getattr(torch.backends.quantized, "supported_engines", [])
    if "qnnpack" not in supported:
        raise RuntimeError(
            f"QNNPACK not available in this build. supported_engines={supported}"
        )
    torch.backends.quantized.engine = "qnnpack"
    if torch.backends.quantized.engine != "qnnpack":
        raise RuntimeError(
            f"Failed to activate qnnpack. current_engine={torch.backends.quantized.engine}"
        )


def build_dynamic_int8_engine(model_name: str = "distilgpt2") -> RuntimeEngine:
    engine = RuntimeEngine(model_name=model_name)

    if engine.adapter.device != "cpu":
        raise RuntimeError("Dynamic int8 quantization path is intended for CPU runs.")

    _ensure_qengine()

    quantized_model = torch.quantization.quantize_dynamic(
        engine.adapter.model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    engine.adapter.model = quantized_model
    engine.adapter.model.eval()
    return engine


def quantization_compare(
    model_name: str = "distilgpt2",
    prompt: str = "Explain why KV-cache improves autoregressive decoding efficiency.",
    max_new_tokens: int = 20,
):
    base_engine = RuntimeEngine(model_name=model_name)
    req_base = InferenceRequest(request_id="fp32", prompt=prompt, max_new_tokens=max_new_tokens)
    base_state = base_engine.run_kv_single(req_base)

    result = {
        "fp32": summarize_states("fp32_kv_single", [base_state]),
        "fp32_text": base_engine.adapter.decode_tokens(base_state.generated_ids),
    }

    try:
        q_engine = build_dynamic_int8_engine(model_name=model_name)
        req_q = InferenceRequest(request_id="int8", prompt=prompt, max_new_tokens=max_new_tokens)
        q_state = q_engine.run_kv_single(req_q)

        result["int8_dynamic"] = summarize_states("int8_dynamic_kv_single", [q_state])
        result["int8_text"] = q_engine.adapter.decode_tokens(q_state.generated_ids)
        result["quantization_status"] = "ok"
        result["active_qengine"] = torch.backends.quantized.engine
    except Exception as e:
        result["quantization_status"] = "skipped"
        result["quantization_error"] = str(e)
        result["active_qengine"] = torch.backends.quantized.engine

    return result