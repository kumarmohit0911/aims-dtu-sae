from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_lm(model_name: str, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.eval()
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    return model, tokenizer


def get_submodule(model: torch.nn.Module, dotted_name: str) -> torch.nn.Module:
    module: torch.nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


@contextmanager
def capture_module_output(
    model: torch.nn.Module,
    module_name: str,
    storage: list[torch.Tensor],
) -> Iterator[None]:
    module = get_submodule(model, module_name)

    def hook(_, __, output):
        tensor = output[0] if isinstance(output, tuple) else output
        storage.append(tensor.detach())

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
