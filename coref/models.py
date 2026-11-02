import os

# bitsandbytes calls torch.compile while TransformerLens imports. Disable
# Dynamo before importing transformer_lens so direct coref.models imports are
# as robust as the script entrypoints that set this environment variable.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformer_lens.HookedTransformer as HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name
from functools import cache


LLAMA_PATH = os.environ["LLAMA_WEIGHTS"] if "LLAMA_WEIGHTS" in os.environ else None
HF_PATH = os.environ["HF_CACHE"] if "HF_CACHE" in os.environ else None
TULU_PATH = os.environ["TULU_WEIGHTS"] if "TULU_WEIGHTS" in os.environ else None
DEFAULT_LLAMA_1B_SOURCE = "meta-llama/Llama-3.2-1B"
DEFAULT_LLAMA2_7B_CHAT_SOURCE = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_LLAMA2_13B_CHAT_SOURCE = "meta-llama/Llama-2-13b-chat-hf"
DEFAULT_LLAMA_8B_SOURCE = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_GEMMA_SOURCE = "google/gemma-2-9b-it"
LLAMA_TOKENIZER_SOURCE_ENV = "COREF_LLAMA_TOKENIZER_SOURCE"

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
]
LLAMA_MODELS = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
    "llama-1b-hf",
    "llama-2-7b-chat",
    "llama-2-7b-chat-hf",
    "llama2-7b-chat",
    "llama2-7b-chat-hf",
    "llama-2-13b-chat",
    "llama-2-13b-chat-hf",
    "llama2-13b-chat",
    "llama2-13b-chat-hf",
    "llama-8b-hf",
    DEFAULT_LLAMA_1B_SOURCE,
    DEFAULT_LLAMA2_7B_CHAT_SOURCE,
    DEFAULT_LLAMA2_13B_CHAT_SOURCE,
    DEFAULT_LLAMA_8B_SOURCE,
]
GEMMA_MODELS = [
    "gemma2-9b-it",
]
TULU_MODELS = [
    "tulu-7b",
    "tulu-13b",
    "tulu-30b",
    "tulu-65b",
]


def resolve_llama_source(model_tag=None):
    if model_tag in {None, "llama-1b-hf", DEFAULT_LLAMA_1B_SOURCE}:
        return DEFAULT_LLAMA_1B_SOURCE

    if model_tag in {"llama-8b-hf", DEFAULT_LLAMA_8B_SOURCE}:
        return DEFAULT_LLAMA_8B_SOURCE

    if model_tag in {
        "llama-2-7b-chat",
        "llama-2-7b-chat-hf",
        "llama2-7b-chat",
        "llama2-7b-chat-hf",
        DEFAULT_LLAMA2_7B_CHAT_SOURCE,
    }:
        return DEFAULT_LLAMA2_7B_CHAT_SOURCE

    if model_tag in {
        "llama-2-13b-chat",
        "llama-2-13b-chat-hf",
        "llama2-13b-chat",
        "llama2-13b-chat-hf",
        DEFAULT_LLAMA2_13B_CHAT_SOURCE,
    }:
        return DEFAULT_LLAMA2_13B_CHAT_SOURCE

    if model_tag in LLAMA_MODELS:
        if LLAMA_PATH is None:
            raise ValueError(
                "LLAMA_WEIGHTS is not set, so local LLaMA weights cannot be loaded."
            )
        llama_size = re.match(r"llama-(\d+)b-hf", model_tag).group(1)
        return os.path.join(LLAMA_PATH, f"{llama_size}B")

    env_source = os.environ.get(LLAMA_TOKENIZER_SOURCE_ENV)
    if env_source:
        return env_source

    if LLAMA_PATH is not None:
        return os.path.join(LLAMA_PATH, "7B")

    return DEFAULT_LLAMA_1B_SOURCE


def resolve_gemma_source(model_tag=None):
    if model_tag in {None, "gemma2-9b-it"}:
        return DEFAULT_GEMMA_SOURCE
    if model_tag in GEMMA_MODELS:
        return model_tag
    if model_tag is not None and "gemma" in model_tag.lower():
        return model_tag
    return DEFAULT_GEMMA_SOURCE


def fetch_model(model_tag, device="cuda", num_devices=1, dtype=None):
    model_tag_lower = model_tag.lower()
    if dtype is None:
        if (
            "gemma" in model_tag_lower
            or "qwen" in model_tag_lower
            or "llama" in model_tag_lower
        ) and torch.cuda.is_available():
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            dtype = torch.float16 if '65b' in model_tag else torch.float32
    print(f'Loading {model_tag} in precision {dtype}')
    torch.set_grad_enabled(False)
    if model_tag in PYTHIA_MODELS:
        hf_model = AutoModelForCausalLM.from_pretrained(
            get_official_model_name(model_tag),
            **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            low_cpu_mem_usage=True,
            torch_dtype=dtype
        )
        model = HookedTransformer.from_pretrained(
            model_tag,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            hf_model=hf_model,
            dtype=dtype
        )
    elif model_tag in LLAMA_MODELS:
        if model_tag in {"llama-1b-hf", DEFAULT_LLAMA_1B_SOURCE}:
            tl_model_name = DEFAULT_LLAMA_1B_SOURCE
            model_path = DEFAULT_LLAMA_1B_SOURCE
        elif model_tag in {"llama-8b-hf", DEFAULT_LLAMA_8B_SOURCE}:
            tl_model_name = DEFAULT_LLAMA_8B_SOURCE
            model_path = DEFAULT_LLAMA_8B_SOURCE
        else:
            tl_model_name = model_tag
            model_path = resolve_llama_source(model_tag)

        os.environ[LLAMA_TOKENIZER_SOURCE_ENV] = model_path

        tokenizer = get_llama_tokenizer(model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model = HookedTransformer.from_pretrained(
            tl_model_name,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype
        )
        model.tokenizer = tokenizer
        if model.tokenizer.pad_token is None and model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
    elif model_tag in GEMMA_MODELS:
        model_path = resolve_gemma_source(model_tag)
        tokenizer = get_gemma_tokenizer(model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        model = HookedTransformer.from_pretrained(
            model_path,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype,
        )
        model.tokenizer = tokenizer
        if model.tokenizer.pad_token is None and model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
    elif model_tag in TULU_MODELS:
        model_size = re.match(r"tulu-(\d+)b", model_tag).group(1)
        llama_tag = f"llama-{model_size}b-hf"
        model_path = os.path.join(TULU_PATH, f"tulu-merged-{model_size}b")
        os.environ[LLAMA_TOKENIZER_SOURCE_ENV] = model_path
        # probably okay to use llama tokenizer instead of tulu tokenizer
        # should be just a difference of a padding token at the end
        tokenizer = get_llama_tokenizer(model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=dtype
        )
        model = HookedTransformer.from_pretrained(
            llama_tag,
            hf_model=hf_model,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype
        )
        model.tokenizer = tokenizer
        # the new model already has this i think
        # model.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        tokenizer_kwargs = dict(cache_dir=HF_PATH) if HF_PATH is not None else {}
        model_kwargs = {
            **tokenizer_kwargs,
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
        }
        if "qwen" in model_tag_lower:
            tokenizer_kwargs["trust_remote_code"] = True
            model_kwargs["trust_remote_code"] = True

        tl_kwargs = {}
        tokenizer = AutoTokenizer.from_pretrained(model_tag, **tokenizer_kwargs)
        hf_model = AutoModelForCausalLM.from_pretrained(model_tag, **model_kwargs)
        if "qwen" in model_tag_lower:
            tl_kwargs["trust_remote_code"] = True
        model = HookedTransformer.from_pretrained(
            model_tag,
            hf_model=hf_model,
            tokenizer=tokenizer,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            refactor_factored_attn_matrices=False,
            fold_value_biases=False,
            device=device,
            n_devices=num_devices,
            dtype=dtype,
            **tl_kwargs,
        )
        model.tokenizer = tokenizer
        if model.tokenizer.pad_token is None and model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

    if device is not None and num_devices == 1:
        # TransformerLens' internal load path can move modules to a generic
        # "best available" CUDA device, which may ignore an explicit index like
        # cuda:2. Force a final full-model move so parameters and input tokens
        # agree on the same device in notebook use.
        model = model.to(str(device), print_details=False)
    model.eval()
    model.model_tag = model_tag
    return model

@cache
def get_llama_tokenizer(source=None):
    if source is None:
        source = resolve_llama_source()
    return AutoTokenizer.from_pretrained(
        source,
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
    )


@cache
def get_gemma_tokenizer(source=None):
    if source is None:
        source = resolve_gemma_source()
    return AutoTokenizer.from_pretrained(
        source,
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
    )
@cache
def get_pythia_tokenizer():
    return AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-2.8b",
        **(dict(cache_dir=HF_PATH) if HF_PATH is not None else {}),
    )
