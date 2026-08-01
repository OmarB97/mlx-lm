# Copyright © 2025 Apple Inc.

"""
Run with:

```
mlx.launch \
    --backend jaccl \
    --env MLX_METAL_FAST_SYNCH=1 \
    --hostfile /path/to/hosts.json \
    /path/to/sharded_generate.py \
    --prompt 'Hello world'
```

For more information on running distributed programs with MLX see the documentation:

https://ml-explore.github.io/mlx/build/html/usage/distributed.html .
"""

import argparse

import mlx.core as mx

from mlx_lm import stream_generate
from mlx_lm.utils import sharded_load

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM distributed inference example")
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.3-70B-Instruct-4bit",
        help="HF repo or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Write a quicksort in C++.",
        help="Message to be processed by the model ('-' reads from stdin)",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use pipelining instead of tensor parallelism",
    )
    args = parser.parse_args()

    group = mx.distributed.init()
    rank = group.rank()
    pipeline_group = group if args.pipeline else None
    tensor_group = group if not args.pipeline else None

    def rprint(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    import sys as _sys, time as _time

    def _phase(msg):
        print(f"[rank {rank}] {_time.strftime('%H:%M:%S')} {msg}", file=_sys.stderr, flush=True)

    import os as _os
    if (_cap := _os.environ.get("MLX_DS4_MEM_LIMIT_GB")):
        import mlx.core as _mx
        _mx.set_memory_limit(int(float(_cap) * 2**30))
        _phase(f"memory limit set to {_cap} GiB")

    _phase("sharded_load: begin")
    model, tokenizer = sharded_load(args.model, pipeline_group, tensor_group)
    _phase(
        "sharded_load: done — active "
        f"{mx.get_active_memory()/2**30:.1f} GiB, peak {mx.get_peak_memory()/2**30:.1f} GiB, "
        f"cache {mx.get_cache_memory()/2**30:.1f} GiB"
    )

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    _phase("generate: begin")
    for response in stream_generate(
        model, tokenizer, prompt, max_tokens=args.max_tokens
    ):
        rprint(response.text, end="", flush=True)

    rprint()
    rprint("=" * 10)
    rprint(
        f"Prompt: {response.prompt_tokens} tokens, "
        f"{response.prompt_tps:.3f} tokens-per-sec"
    )
    rprint(
        f"Generation: {response.generation_tokens} tokens, "
        f"{response.generation_tps:.3f} tokens-per-sec"
    )
    rprint(f"Peak memory: {response.peak_memory:.3f} GB")
