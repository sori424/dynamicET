"""Integrity checks shared by score producers and consumers."""

import hashlib
import json
from pathlib import Path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_score_artifact(path: Path) -> None:
    metadata_path = path.parent / "path_patching_metadata.json"
    if not metadata_path.is_file():
        return  # External/legacy score tensors have no producer manifest.
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "run_config" not in metadata:
        return
    saved = next((stage for stage in metadata.get("stages", {}).values()
                  if stage.get("file") == path.name), None)
    expected = saved.get("sha256") if saved else None
    if path.name == "struct_reader.pt":
        saved = metadata.get("stages", {}).get("struct_reader_swap_target", {})
        expected = saved.get("legacy_sha256")
    if expected is None:
        raise ValueError(f"Score tensor is not complete in the current run: {path}. "
                         "Finish run_path_patching.py before consuming its outputs.")
    if file_digest(path) != expected:
        raise ValueError(f"Saved score tensor changed: {path}. Recompute path patching.")
