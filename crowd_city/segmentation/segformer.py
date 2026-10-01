"""SegFormer Cityscapes inference wrapper used to locate road and footpath.

The model is loaded lazily and only once per process. Frames arrive already
scaled to the network input size by ffmpeg, so no resizing happens here and the
returned label map keeps a fixed, known geometry: normalised bounding-box
coordinates map linearly onto it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# Any nvidia/segformer-b*-finetuned-cityscapes checkpoint is a drop-in
# replacement; b4 and b5 are more accurate and correspondingly slower. The
# smaller b0 variant exists as an escape hatch when segmentation throughput,
# rather than accuracy, is the binding constraint.
DEFAULT_MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
DEFAULT_INPUT_WIDTH = 1024
DEFAULT_INPUT_HEIGHT = 512
DEFAULT_BATCH_SIZE = 8


def resolve_device(requested: Optional[str] = None) -> str:
    """Return the torch device to run segmentation on."""
    import torch

    choice = str(requested or "auto").strip().lower()
    if choice not in {"", "auto"}:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SurfaceSegmenter:
    """Batch SegFormer inference returning Cityscapes trainId label maps."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_width: int = DEFAULT_INPUT_WIDTH,
        input_height: int = DEFAULT_INPUT_HEIGHT,
    ) -> None:
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.device = resolve_device(device)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        logger.info(
            f"Loading segmentation model {self.model_name} on device {self.device}."
        )
        self._processor = SegformerImageProcessor.from_pretrained(self.model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)
        self._model = model

        # A warning here is far cheaper than discovering a silent class-order
        # mismatch after a full run.
        labels = getattr(model.config, "id2label", {}) or {}
        if len(labels) != 19:
            logger.warning(
                f"Segmentation model {self.model_name} reports {len(labels)} classes; "
                "CROWD surface mapping assumes the 19 Cityscapes evaluation classes."
            )
        del torch

    @property
    def model_identifier(self) -> str:
        """Return the identifier recorded in the segmentation manifest."""
        return f"{self.model_name}@{self.input_width}x{self.input_height}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def segment(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the trainId map and per-pixel confidence for RGB ``frames``.

        ``frames`` is an ``(n, height, width, 3)`` uint8 array already scaled to
        the network input size. The returned maps are the raw SegFormer output
        grid, which is a quarter of the input resolution in each axis; callers
        address it with normalised coordinates so the exact size never matters.
        """
        import torch

        if frames.size == 0:
            return (
                np.zeros((0, 0, 0), dtype=np.int16),
                np.zeros((0, 0, 0), dtype=np.float32),
            )

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        label_batches = []
        confidence_batches = []

        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                chunk = frames[start:start + self.batch_size]
                inputs = self._processor(
                    images=list(chunk),
                    do_resize=False,
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(self.device)
                logits = self._model(pixel_values=pixel_values).logits
                probabilities = torch.softmax(logits.float(), dim=1)
                confidence, labels = probabilities.max(dim=1)
                label_batches.append(labels.to("cpu").numpy().astype(np.int16))
                confidence_batches.append(
                    confidence.to("cpu").numpy().astype(np.float32)
                )

        return (
            np.concatenate(label_batches, axis=0),
            np.concatenate(confidence_batches, axis=0),
        )


def segmentation_is_available() -> bool:
    """Return whether the optional segmentation dependencies are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        logger.warning(f"Segmentation dependencies are unavailable: {error}")
        return False
    return True


def offline_mode_enabled() -> bool:
    """Return whether Hugging Face downloads are disabled for this process."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = str(os.environ.get(name, "")).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


# ----------------------------------------------------------------------
# Restartable GPU worker
# ----------------------------------------------------------------------

# Frames sent to the worker per message, so a long window never has to be
# pickled in one piece.
WORKER_CHUNK_FRAMES = 64
# A chunk that takes longer than this is treated as a hung GPU. Normal chunks
# take well under ten seconds.
WORKER_CALL_TIMEOUT_SECONDS = 300.0
# Restarts allowed in one process before the failure is treated as permanent.
MAXIMUM_WORKER_RESTARTS = 25


def _segmenter_worker(read_fd: int, write_fd: int, settings_json: str) -> None:
    """Child process: run SurfaceSegmenter on the frames the parent sends.

    Started as a separate interpreter (see IsolatedSurfaceSegmenter._start)
    rather than through multiprocessing, whose spawn method would re-run the
    parent's main script, i.e. a second analysis.py.
    """
    import json
    from multiprocessing.connection import Connection

    receive = Connection(read_fd, writable=False)
    reply = Connection(write_fd, readable=False)
    segmenter = SurfaceSegmenter(**json.loads(settings_json))
    while True:
        try:
            message = receive.recv()
        except EOFError:
            return
        if message is None:
            return
        try:
            reply.send(("ok", segmenter.segment(message)))
        except Exception as error:
            reply.send(("error", f"{type(error).__name__}: {error}"))
            if "CUDA" in str(error):
                # The CUDA context is now unusable; let the parent start afresh.
                return


class _WorkerConnection:
    """The two one-way pipes to the worker, used like one duplex Connection."""

    def __init__(self, sender, receiver) -> None:
        self._sender, self._receiver = sender, receiver

    def send(self, value) -> None:
        self._sender.send(value)

    def poll(self, timeout: float) -> bool:
        return self._receiver.poll(timeout)

    def recv(self):
        return self._receiver.recv()

    def close(self) -> None:
        self._sender.close()
        self._receiver.close()


class IsolatedSurfaceSegmenter:
    """SurfaceSegmenter running in a child process that is restarted on GPU failure.

    An NVIDIA watchdog timeout (Xid 8) leaves the CUDA context of the process
    that owned it broken for good, so with the model in the analysis process
    one such lock-up ended a multi-day run. Here only the child owns the GPU:
    when it reports a CUDA error, dies, or stops answering, it is killed and a
    new one is started, and the chunk is tried once more. Labels do not depend
    on which process produced them, so a retried chunk gives the same result.
    The interface matches SurfaceSegmenter.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_width: int = DEFAULT_INPUT_WIDTH,
        input_height: int = DEFAULT_INPUT_HEIGHT,
    ) -> None:
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.device = str(device or "auto")
        self._settings = dict(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            input_width=self.input_width,
            input_height=self.input_height,
        )
        self._process = None
        self._connection = None
        self.restarts = 0

    @property
    def model_identifier(self) -> str:
        return f"{self.model_name}@{self.input_width}x{self.input_height}"

    def _start(self) -> None:
        import json
        import subprocess
        import sys
        from multiprocessing.connection import Connection

        to_child_read, to_child_write = os.pipe()
        from_child_read, from_child_write = os.pipe()
        repository_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (repository_root, environment.get("PYTHONPATH", "")) if part
        )
        code = (
            "import sys; from crowd_city.segmentation.segformer import _segmenter_worker; "
            "_segmenter_worker(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(to_child_read), str(from_child_write), json.dumps(self._settings)],
            pass_fds=(to_child_read, from_child_write),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
        )
        os.close(to_child_read)
        os.close(from_child_write)
        self._process = process
        self._connection = _WorkerConnection(
            Connection(to_child_write, readable=False), Connection(from_child_read, writable=False)
        )

    def close(self) -> None:
        """Stop the worker; the next call starts a new one."""
        process, connection = self._process, self._connection
        self._process = self._connection = None
        if connection is not None:
            try:
                connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            connection.close()
        if process is not None:
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait(timeout=10)

    def _kill(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=10)
        if self._connection is not None:
            self._connection.close()
        self._process = self._connection = None

    def _call(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._process is None or self._process.poll() is not None:
            self._start()
        assert self._connection is not None
        try:
            self._connection.send(frames)
        except (BrokenPipeError, OSError):
            raise RuntimeError("CUDA error: the segmentation worker exited unexpectedly") from None
        if not self._connection.poll(WORKER_CALL_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"CUDA error: the segmentation worker did not answer within {WORKER_CALL_TIMEOUT_SECONDS:.0f}s"
            )
        try:
            status, payload = self._connection.recv()
        except EOFError:
            raise RuntimeError("CUDA error: the segmentation worker exited unexpectedly") from None
        if status != "ok":
            raise RuntimeError(payload)
        return payload

    def _call_with_restart(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self._call(frames)
        except RuntimeError as error:
            if "CUDA" not in str(error):
                raise
            self._kill()
            self.restarts += 1
            if self.restarts > MAXIMUM_WORKER_RESTARTS:
                raise RuntimeError(
                    f"CUDA error: the GPU failed {self.restarts} times in this run, last with: {error}"
                ) from error
            logger.warning(
                f"The segmentation GPU worker failed ({error}); restarting it "
                f"(restart {self.restarts} of at most {MAXIMUM_WORKER_RESTARTS}) and retrying."
            )
            return self._call(frames)

    def segment(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Same as SurfaceSegmenter.segment, computed in the worker process."""
        if frames.size == 0:
            return (
                np.zeros((0, 0, 0), dtype=np.int16),
                np.zeros((0, 0, 0), dtype=np.float32),
            )
        labels, confidence = [], []
        for start in range(0, len(frames), WORKER_CHUNK_FRAMES):
            chunk_labels, chunk_confidence = self._call_with_restart(frames[start:start + WORKER_CHUNK_FRAMES])
            labels.append(chunk_labels)
            confidence.append(chunk_confidence)
        return np.concatenate(labels, axis=0), np.concatenate(confidence, axis=0)
