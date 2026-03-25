import argparse
import json

from src.benchmarks.benchmark_suite import benchmark_suite
from src.benchmarks.scaling import prompt_scaling_benchmark
from src.benchmarks.quantization import quantization_compare
from src.benchmarks.compile_compare import compile_compare
from src.runtime.correctness import validate_single_vs_kv
from src.runtime.speculative import SpeculativeDecoder
from src.core.types import InferenceRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--n_requests", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=20)

    parser.add_argument("--check", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--compile_compare", action="store_true")
    parser.add_argument("--speculative", action="store_true")

    args = parser.parse_args()

    if args.check:
        out = validate_single_vs_kv(model_name=args.model)
        print("\n=== Correctness Check ===")
        print(json.dumps(out, indent=2))
        return

    if args.scaling:
        df = prompt_scaling_benchmark(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n=== Prompt Scaling Benchmark ===")
        print(df.round(4).to_string(index=False))
        return

    if args.quant:
        out = quantization_compare(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n=== Quantization Compare ===")
        print(json.dumps(out, indent=2))
        return

    if args.compile_compare:
        out = compile_compare(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n=== torch.compile Compare ===")
        print(json.dumps(out, indent=2))
        return

    if args.speculative:
        decoder = SpeculativeDecoder(
            draft_model_name="sshleifer/tiny-gpt2",
            target_model_name=args.model,
        )
        req = InferenceRequest(
            request_id="speculative_demo",
            prompt="Explain speculative decoding in simple terms.",
            max_new_tokens=args.max_new_tokens,
        )
        out = decoder.generate(req, draft_steps=4)
        print("\n=== Speculative Decoding Demo ===")
        print(json.dumps(out, indent=2))
        return

    df, _ = benchmark_suite(
        model_name=args.model,
        n_requests=args.n_requests,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n=== Benchmark Summary ===")
    print(df.round(4).to_string(index=False))
    print("\nSaved outputs to outputs/")


if __name__ == "__main__":
    main()