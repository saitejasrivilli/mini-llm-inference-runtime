#!/bin/bash
set -e

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py --model distilgpt2 --n_requests 6 --max_new_tokens 20
python main.py --model sshleifer/tiny-gpt2 --check
python main.py --model distilgpt2 --scaling --max_new_tokens 12

python main.py --model distilgpt2 --quant --max_new_tokens 20 || echo "Skipping quantization on this platform"

python main.py --model distilgpt2 --compile_compare --max_new_tokens 20 || echo "Skipping torch.compile compare on this platform"

python main.py --model distilgpt2 --speculative --max_new_tokens 12