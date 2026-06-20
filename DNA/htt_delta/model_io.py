"""Local CENO loading helpers.

The CENO model is derived from NVIDIA's Nemotron-H (Apache-2.0). The bundled
custom Transformers remote code (`configuration_ceno.py`, `modeling_ceno.py`)
is a rename of the upstream Nemotron-H implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoConfig


def import_local_ceno(model_code_dir: str | Path):
    model_code_dir = Path(model_code_dir).resolve()
    required = [
        "configuration_ceno.py",
        "modeling_ceno.py",
        "ceno_tokenizer.py",
    ]
    missing = [name for name in required if not (model_code_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required CENO files in {model_code_dir}: {missing}"
        )

    sys.path.insert(0, str(model_code_dir))
    from configuration_ceno import CENOConfig
    from ceno_tokenizer import CENOCharLevelTokenizer
    from modeling_ceno import CENOForCausalLM

    AutoConfig.register("ceno", CENOConfig)
    return CENOConfig, CENOForCausalLM, CENOCharLevelTokenizer


def load_model_and_tokenizer(
    model_path: str | Path,
    model_code_dir: str | Path,
    device: torch.device,
):
    CENOConfig, CENOForCausalLM, CENOCharLevelTokenizer = import_local_ceno(
        model_code_dir
    )
    tokenizer = CENOCharLevelTokenizer(vocab_size=512)
    config = CENOConfig.from_pretrained(model_path)
    config.use_mamba_kernels = False
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = CENOForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer, config

