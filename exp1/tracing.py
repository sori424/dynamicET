import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fire
import numpy as np

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch

from tracing_utils import (
    build_tracing_dataset,
    find_correct_samples,
    load_model,
    normalize_trace_category,
    resolve_query_trace_mode,
    resolve_trace_modes,
    run_tracing_experiment,
)


def configure_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REPO_ROOT = configure_repo_root()
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tracing_entity"
DEFAULT_DEVICE_ID = "0"
SYSTEM_PROMPT = "Respond with a single word only. No punctuation, no explanation."
ANSWER_PREFILL_RE = re.compile(r"\s*Answer:\s*$", re.IGNORECASE)
MODEL_NAME_ALIASES = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "llama-8b-hf"
}
PROMPT_FORMATS = {
    "raw",
    "auto",
    "chat-template",
    "chat-prefill",
    "gemma-it-prefill",
    "llama-instruct-prefill",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tracing")

from exp1.prompt_formatting import (
    resolve_model_name,
    infer_prompt_format,
    resolve_prompt_format,
    _apply_chat_template,
    _split_assistant_prefill,
    _tokenizer_has_token,
    _format_llama_prefill_fallback,
    _apply_chat_prefill_template,
    format_prompt_for_model,
    _ensure_pad_token,
    _last_token_indices,
    _last_token_indices_from_attention_mask,
    _encode_prompt_texts,
    _answer_words_from_tokens,
    _first_answer_token_id,
    _format_answer_tokens,
    apply_prompt_format_to_dataset,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Tracer:
    entity_type: str
    model_name: str
    vocab_tag: str
    split: str
    results_dir: Path
    num_samples: int
    batch_size: int
    tracing_batch_size: int
    oversample_factor: int
    num_entities: int
    prompt_id_start: int
    query_name: int
    swap_box_a: Optional[int]
    swap_box_b: Optional[int]
    query_trace_mode: str
    names_trace_mode: str
    object_trace_mode: str
    start_token: Optional[int]
    end_token: int
    start_layer: int
    layer_step: int
    num_devices: int
    device_id: str
    prompt_format: str
    verbose: bool

    def run(self) -> Path:
        trace_category = normalize_trace_category(self.entity_type, self.vocab_tag)
        names_trace_mode, object_trace_mode = resolve_trace_modes(
            trace_category,
            names_trace_mode=self.names_trace_mode,
            object_trace_mode=self.object_trace_mode,
        )
        query_trace_mode = resolve_query_trace_mode(self.query_trace_mode)
        device_id = str(self.device_id)
        trace_mode_suffix = ""
        if (
            self.vocab_tag == "BOXES"
            and trace_category == "names"
            and names_trace_mode != "context_only"
        ):
            trace_mode_suffix = f"_{names_trace_mode}"
        elif (
            self.vocab_tag == "BOXES"
            and trace_category == "countries"
            and object_trace_mode != "target_box"
        ):
            trace_mode_suffix = f"_{object_trace_mode}"
        query_trace_mode_suffix = ""
        if query_trace_mode != "same_query":
            query_trace_mode_suffix = f"_query_{query_trace_mode}"
        swap_pair_suffix = ""
        if self.swap_box_a is not None and self.swap_box_b is not None:
            swap_pair_suffix = f"_swap{self.swap_box_a}-{self.swap_box_b}"
        results_path = (
            self.results_dir
            / (
                f"{self.vocab_tag.lower()}_{trace_category}{trace_mode_suffix}"
                f"{query_trace_mode_suffix}"
                f"{swap_pair_suffix}"
                f"_n{self.num_entities}_q{self.query_name}.json"
            )
        )

        logger.info(f"Trace type: {self.entity_type} -> {trace_category}")
        resolved_model_name = resolve_model_name(self.model_name)
        logger.info(f"Model: {self.model_name}")
        if resolved_model_name != self.model_name:
            logger.info(f"Resolved model: {resolved_model_name}")
        logger.info(f"Vocab tag: {self.vocab_tag}")
        logger.info(f"Split: {self.split}")
        logger.info(f"Num entities: {self.num_entities}")
        logger.info(
            "Swap pair: "
            + (
                f"({self.swap_box_a}, {self.swap_box_b})"
                if self.swap_box_a is not None and self.swap_box_b is not None
                else "auto(query_name,last_non_query)"
            )
        )
        logger.info(f"Query trace mode: {query_trace_mode}")
        logger.info(f"Names trace mode: {names_trace_mode}")
        logger.info(f"Object trace mode: {object_trace_mode}")
        logger.info(f"Results path: {results_path}")

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        model = load_model(resolved_model_name, num_devices=self.num_devices)
        logger.info(f"Loaded model: {model.model_tag}")
        prompt_format = resolve_prompt_format(model.model_tag, self.prompt_format)
        logger.info(f"Prompt format: {prompt_format}")

        vocab, dataset, metadata = build_tracing_dataset(
            vocab_tag=self.vocab_tag,
            model_tag=model.model_tag,
            split=self.split,
            tokenizer=model.tokenizer,
            trace_type=self.entity_type,
            num_samples=self.num_samples,
            num_entities=self.num_entities,
            prompt_id_start=self.prompt_id_start,
            query_name=self.query_name,
            swap_box_a=self.swap_box_a,
            swap_box_b=self.swap_box_b,
            names_trace_mode=names_trace_mode,
            object_trace_mode=object_trace_mode,
            query_trace_mode=query_trace_mode,
            oversample_factor=self.oversample_factor,
        )
        dataset = apply_prompt_format_to_dataset(
            dataset=dataset,
            tokenizer=model.tokenizer,
            prompt_format=prompt_format,
        )
        metadata.update(
            {
                "prompt_format": prompt_format,
                "requested_prompt_format": self.prompt_format,
            }
        )

        logger.info(
            f"Built {metadata['num_candidates']} candidate prompt pairs "
            f"for {metadata['trace_category']} tracing."
        )
        print(
            f"trace_type / {self.entity_type}: trace_category={trace_category}",
            file=sys.stderr,
        )
        position_debug = metadata.get("position_debug", {})
        for prompt_side in ["clean", "source"]:
            for row in position_debug.get(prompt_side, {}).get(trace_category, []):
                print(
                    f"{prompt_side} / {trace_category}[{row['index']}]: "
                    f"pos={row['pos']}, tok_id={row['tok_id']}, "
                    f"decoded={row['decoded']!r}, stripped={row['stripped']!r}",
                    file=sys.stderr,
                )
        logger.info("Finding prompt pairs where the source prompt predicts the traced answer...")
        correct_indices = find_correct_samples(
            model=model,
            dataset=dataset,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            verbose=self.verbose,
        )
        if len(correct_indices) < self.num_samples:
            raise ValueError(
                f"Only found {len(correct_indices)} source-correct samples, "
                f"but requested {self.num_samples}. Increase oversample_factor or reduce num_samples."
            )

        filtered_dataset = dataset.select(correct_indices[: self.num_samples])
        reference_clean_query_name = int(
            filtered_dataset.payload["clean_query_names"][0].item()
        )
        reference_source_query_name = int(
            filtered_dataset.payload["source_query_names"][0].item()
        )
        metadata.update(
            {
                "model_name": self.model_name,
                "model_tag": model.model_tag,
                "num_samples": self.num_samples,
                "num_entities": self.num_entities,
                "requested_swap_box_a": self.swap_box_a,
                "requested_swap_box_b": self.swap_box_b,
                "batch_size": self.batch_size,
                "tracing_batch_size": self.tracing_batch_size,
                "oversample_factor": self.oversample_factor,
                "names_trace_mode": names_trace_mode,
                "object_trace_mode": object_trace_mode,
                "query_trace_mode": query_trace_mode,
                "start_token": self.start_token,
                "end_token": self.end_token,
                "start_layer": self.start_layer,
                "layer_step": self.layer_step,
                "num_devices": self.num_devices,
                "device_id": device_id,
                "reference_clean_query_name": reference_clean_query_name,
                "reference_source_query_name": reference_source_query_name,
                "reference_clean_prompt": filtered_dataset.payload["clean_prompts"][0],
                "reference_source_prompt": filtered_dataset.payload["source_prompts"][0],
                "reference_clean_answer": vocab.tokenizer.decode(
                    [
                        int(
                            filtered_dataset.payload["clean_answer_tokens"][
                                0, reference_clean_query_name
                            ].item()
                        )
                    ]
                ).strip(),
                "reference_target_answer": vocab.tokenizer.decode(
                    [
                        int(
                            filtered_dataset.payload["source_answer_tokens"][
                                0, reference_source_query_name
                            ].item()
                        )
                    ]
                ).strip(),
                "reference_source_answer": vocab.tokenizer.decode(
                    [
                        int(
                            filtered_dataset.payload["source_answer_tokens"][
                                0, reference_source_query_name
                            ].item()
                        )
                    ]
                ).strip(),
            }
        )
        print(
            f"reference_clean_prompt: {metadata['reference_clean_prompt']!r}",
            file=sys.stderr,
        )
        print(
            f"reference_source_prompt: {metadata['reference_source_prompt']!r}",
            file=sys.stderr,
        )

        logger.info("Running residual tracing experiment...")
        run_tracing_experiment(
            model=model,
            dataset=filtered_dataset,
            tracing_batch_size=self.tracing_batch_size,
            results_path=results_path,
            metadata=metadata,
            start_token=self.start_token,
            end_token=self.end_token,
            start_layer=self.start_layer,
            layer_step=self.layer_step,
            verbose=self.verbose,
        )
        logger.info(f"Saved tracing results to {results_path}")
        return results_path


def main(
    entity_type: str,
    model_name: str,
    results_dir: str = str(DEFAULT_RESULTS_DIR),
    vocab_tag: str = "BOXES",
    split: str = "test",
    num_samples: int = 50,
    batch_size: int = 16,
    tracing_batch_size: int = 8,
    oversample_factor: int = 4,
    num_entities: int = 3,
    prompt_id_start: int = 0,
    query_name: int = 0,
    swap_box_a: Optional[int] = None,
    swap_box_b: Optional[int] = None,
    query_trace_mode: str = "same_query",
    names_trace_mode: str = "context_only",
    object_trace_mode: str = "target_box",
    start_token: Optional[int] = None,
    end_token: int = 0,
    start_layer: int = 0,
    layer_step: int = 1,
    num_devices: int = 1,
    device_id=DEFAULT_DEVICE_ID,
    prompt_format: str = "auto",
    verbose: bool = False,
    random_seed: int = 10,
    data_dir: str = None,
    is_remote: bool = False,
):

    set_seed(random_seed)

    results_dir_path = Path(results_dir)
    if not results_dir_path.is_absolute():
        results_dir_path = REPO_ROOT / results_dir_path
    results_dir_path.mkdir(parents=True, exist_ok=True)

    tracer = Tracer(
        entity_type=entity_type,
        model_name=model_name,
        vocab_tag=vocab_tag,
        split=split,
        results_dir=results_dir_path,
        num_samples=num_samples,
        batch_size=batch_size,
        tracing_batch_size=tracing_batch_size,
        oversample_factor=oversample_factor,
        num_entities=num_entities,
        prompt_id_start=prompt_id_start,
        query_name=query_name,
        swap_box_a=swap_box_a,
        swap_box_b=swap_box_b,
        query_trace_mode=query_trace_mode,
        names_trace_mode=names_trace_mode,
        object_trace_mode=object_trace_mode,
        start_token=start_token,
        end_token=end_token,
        start_layer=start_layer,
        layer_step=layer_step,
        num_devices=num_devices,
        device_id=device_id,
        prompt_format=prompt_format,
        verbose=verbose,
    )
    tracer.run()


if __name__ == "__main__":
    fire.Fire(main)
