from __future__ import annotations

from exp2.pp_prompts import (
    CUSTOM_LLAMA_FAMILY_MODELS,
    MODEL_ALIASES,
    _ensure_pad_token,
    device,
)

from coref.models import fetch_model
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
import torch

def _load_custom_llama_family_model(model_source: str) -> Tuple[HookedTransformer, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    _ensure_pad_token(tokenizer)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        "llama-7b-hf",
        hf_model=hf_model,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        device=device,
        dtype=torch.float32,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def _load_goat_model() -> Tuple[HookedTransformer, PreTrainedTokenizerBase]:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading GOAT requires the optional 'peft' package.") from exc

    base_model = "decapoda-research/llama-7b-hf"
    lora_weights = "tiedong/goat-lora-7b"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    _ensure_pad_token(tokenizer)

    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
    )
    hf_model = PeftModel.from_pretrained(
        hf_model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    if hasattr(hf_model, "merge_and_unload"):
        hf_model = hf_model.merge_and_unload()

    model = HookedTransformer.from_pretrained(
        "llama-7b-hf",
        hf_model=hf_model,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        device=device,
        dtype=torch.float32,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def compute_topk_components(
    patching_scores: torch.Tensor, k: int, largest=True, return_values=False
):
    """
    Computes the topk most influential components (i.e. heads) for patching.

    Args:
        patching_scores: patching scores for the components.
        k: number of components to return.
        largest: whether to return the largest or smallest components.
        return_values: whether to return the values of the components or not.
    """

    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices
    top_values = torch.topk(patching_scores.flatten(), k, largest=largest).values

    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1).tolist()

    if return_values:
        return top_components, top_values.tolist()
    return top_components


def compute_prev_query_box_pos(input_ids, last_token_index):
    """
    Computes the position of the previous query box label token.

    Args:
        input_ids: input ids of the example.
        last_token_index: last token index of the example.
    """

    query_box_token = input_ids[last_token_index - 2]
    prev_query_box_token_pos = (
        (input_ids[: last_token_index - 2] == query_box_token).nonzero().item()
    )
    return prev_query_box_token_pos


def get_model_and_tokenizer(model_name: str):
    """
    Loads the model and tokenizer through TransformerLens.

    Args:
        model_name (str): Name of the model to load.
    """

    resolved_name = MODEL_ALIASES.get(model_name, model_name)

    if model_name in CUSTOM_LLAMA_FAMILY_MODELS:
        return _load_custom_llama_family_model(CUSTOM_LLAMA_FAMILY_MODELS[model_name])

    if model_name == "goat":
        return _load_goat_model()

    model = fetch_model(resolved_name, device=str(device))
    tokenizer = model.tokenizer
    _ensure_pad_token(tokenizer)
    return model, tokenizer
