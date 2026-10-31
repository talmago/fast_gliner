"""Load Hub or local checkpoints and construct a Rust runtime."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

from huggingface_hub import snapshot_download

_ExecutionProvider = Optional[Literal["cpu", "cuda"]]


class _FastGLiNERBase(ABC):
    """Shared Hub/local loading for the Python runtimes.

    Concrete runtimes set `_backend` to the PyO3 class that owns the ONNX
    session, implement `from_pretrained`, and resolve files inside the
    checkpoint directory themselves.

    Attributes:
        model: Bound Rust runtime instance.
    """

    _backend = None
    _snapshot_allow_patterns = ("*.json", "*.model", "*.onnx", "onnx/*.onnx")

    def __init__(
        self,
        model_path: str,
        onnx_path: Optional[str] = None,
        execution_provider: _ExecutionProvider = None,
    ):
        """Load a runtime from an on-disk checkpoint.

        Args:
            model_path: Absolute path to the checkpoint directory.
            onnx_path: ONNX graph inside `model_path`. Split-graph runtimes
                pass `None`.
            execution_provider: ONNX Runtime execution provider. `None` uses
                the runtime default (CPU).
        """
        self.model = self._backend(model_path, onnx_path, execution_provider)

    @classmethod
    def _load_pretrained(cls, model_id: str, **kwargs) -> Path:
        """Return a checkpoint directory, downloading from the Hub if needed.

        This only resolves the directory. Callers inspect ONNX files inside
        it.

        Args:
            model_id: Hugging Face repository id or local directory path.
            **kwargs: Extra arguments forwarded to
                `huggingface_hub.snapshot_download`. `allow_patterns`
                defaults to `_snapshot_allow_patterns`.

        Returns:
            Absolute path to the checkpoint directory.

        Raises:
            FileNotFoundError: If `model_id` is an explicit local path that
                is missing, or a file rather than a directory.
        """
        path = Path(model_id).expanduser()
        if path.is_dir():
            return path.resolve()
        if path.exists():
            raise FileNotFoundError(f"Expected a model directory, got a file: {path}")
        if _is_explicit_local_path(model_id, path):
            raise FileNotFoundError(f"Local model directory not found: {path}")

        kwargs.setdefault("allow_patterns", list(cls._snapshot_allow_patterns))
        return Path(snapshot_download(repo_id=model_id, **kwargs)).resolve()

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        """Load a pretrained model from the Hub or a local directory."""


def _is_explicit_local_path(model_id: str, path: Path) -> bool:
    """Return whether `model_id` should never be sent to the Hub.

    Args:
        model_id: Original identifier passed to `from_pretrained`.
        path: `model_id` as an expanded `Path`.

    Returns:
        True for absolute paths and paths that start with `~`, `.`, or a
        filesystem separator.
    """
    return path.is_absolute() or model_id.startswith(("~", ".", "/", "\\"))


def resolve_onnx_filename(model_dir: Path, onnx_path: Optional[str]) -> str:
    """Pick a single ONNX graph inside a checkpoint.

    Looks for the requested path, then `onnx/model.onnx`, then `model.onnx`.
    If none of those exist and the directory contains exactly one `.onnx`
    file, that file is used.

    Args:
        model_dir: Checkpoint directory.
        onnx_path: Preferred ONNX path relative to `model_dir`, or an absolute
            file path. Defaults to `onnx/model.onnx`.

    Returns:
        ONNX path to pass to the Rust runtime.

    Raises:
        FileNotFoundError: If no ONNX graph can be resolved.
    """
    requested = onnx_path or "onnx/model.onnx"
    candidates = [requested, "onnx/model.onnx", "model.onnx"]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        resolved = path if path.is_absolute() else model_dir / path
        if resolved.is_file():
            return candidate if path.is_absolute() else path.as_posix()

    onnx_files = sorted(p for p in model_dir.rglob("*.onnx") if p.is_file())
    if len(onnx_files) == 1:
        return onnx_files[0].relative_to(model_dir).as_posix()

    found = ", ".join(p.relative_to(model_dir).as_posix() for p in onnx_files) or "none"
    raise FileNotFoundError(f"Could not resolve ONNX model inside {model_dir} (looked for {requested}; found {found})")
